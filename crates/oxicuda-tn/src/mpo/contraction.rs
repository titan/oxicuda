//! Contraction of MPOs onto MPSs.

use crate::mpo::mpo::Mpo;
use crate::mps::mps::Mps;
use crate::mps::tensor::MpsTensor;
use crate::mps::truncation::svd_truncate;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// Apply an MPO to an MPS site by site, returning a new MPS with bond dimensions
/// `D' = W * D` (or truncated to `chi_max`).
///
/// The local update at site `s` is:
///   new[(a*W_l), p_out, (b*W_r)] = sum_{p_in} mpo[w_l, p_out, p_in, w_r] * mps[a, p_in, b]
/// followed by a left-to-right SVD sweep with truncation.
pub fn apply_mpo_to_mps(mpo: &Mpo, mps: &Mps, chi_max: usize, tol: f64) -> TnResult<Mps> {
    if mpo.n_sites() != mps.n_sites() {
        return Err(TnError::DimensionMismatch {
            a: mpo.n_sites(),
            b: mps.n_sites(),
        });
    }
    // Step 1: form per-site combined tensors without truncation.
    let mut combined: Vec<MpsTensor> = Vec::with_capacity(mps.n_sites());
    for (mpot, mst) in mpo.site_tensors.iter().zip(mps.site_tensors.iter()) {
        let (w_l, d_out, d_in, w_r) = mpot.shape();
        let (d_l, d_p, d_r) = mst.shape();
        if d_in != d_p {
            return Err(TnError::DimensionMismatch { a: d_in, b: d_p });
        }
        let new_dl = d_l * w_l;
        let new_dr = d_r * w_r;
        let mut data = vec![0.0; new_dl * d_out * new_dr];
        for a in 0..d_l {
            for w_lc in 0..w_l {
                for p_out in 0..d_out {
                    for b in 0..d_r {
                        for w_rc in 0..w_r {
                            let mut acc = 0.0;
                            for p_in in 0..d_p {
                                let mv =
                                    mpot.data[((w_lc * d_out + p_out) * d_in + p_in) * w_r + w_rc];
                                let sv = mst.data[(a * d_p + p_in) * d_r + b];
                                acc += mv * sv;
                            }
                            let new_a = a * w_l + w_lc;
                            let new_b = b * w_r + w_rc;
                            data[(new_a * d_out + p_out) * new_dr + new_b] = acc;
                        }
                    }
                }
            }
        }
        combined.push(MpsTensor::new(new_dl, d_out, new_dr, data)?);
    }
    let mut result = Mps::from_tensors(combined)?;
    // Step 2: left-to-right SVD truncation sweep
    for s in 0..result.n_sites() - 1 {
        let (dl, dp, dr) = result.site_tensors[s].shape();
        let m = dl * dp;
        let svd = svd_jacobi(&result.site_tensors[s].data, m, dr)?;
        let (svd, _) = svd_truncate(svd, chi_max, tol)?;
        let k = svd.k;
        let mut new_left = vec![0.0; dl * dp * k];
        for i in 0..m {
            for j in 0..k {
                new_left[i * k + j] = svd.u[i * k + j];
            }
        }
        result.site_tensors[s] = MpsTensor::new(dl, dp, k, new_left)?;
        // Multiply diag(s) * V^T into next site
        let (dr_old, dp_next, dr_next) = result.site_tensors[s + 1].shape();
        if dr_old != dr {
            return Err(TnError::DimensionMismatch { a: dr_old, b: dr });
        }
        let mut new_next = vec![0.0; k * dp_next * dr_next];
        let next_data = &result.site_tensors[s + 1].data;
        for new_a in 0..k {
            let sv = svd.s[new_a];
            for p in 0..dp_next {
                for c in 0..dr_next {
                    let mut acc = 0.0;
                    for old_a in 0..dr {
                        let vtv = svd.vt[new_a * dr + old_a];
                        let nv = next_data[(old_a * dp_next + p) * dr_next + c];
                        acc += sv * vtv * nv;
                    }
                    new_next[(new_a * dp_next + p) * dr_next + c] = acc;
                }
            }
        }
        result.site_tensors[s + 1] = MpsTensor::new(k, dp_next, dr_next, new_next)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpo::mpo::Mpo;
    use crate::mps::mps::Mps;

    #[test]
    fn identity_mpo_preserves_mps() {
        let local = vec![vec![1.0, 0.0]; 3];
        let mps = Mps::from_product_state(&local).expect("ok");
        let mpo = Mpo::identity(3, 2).expect("ok");
        let out = apply_mpo_to_mps(&mpo, &mps, 4, 1e-12).expect("ok");
        let n_in = mps.norm_squared().expect("ok");
        let n_out = out.norm_squared().expect("ok");
        assert!((n_in - n_out).abs() < 1e-9);
    }
}
