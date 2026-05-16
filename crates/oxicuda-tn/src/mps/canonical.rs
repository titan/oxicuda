//! Canonicalization of an MPS via QR (or SVD) sweeps.
//!
//! ## Left-canonical
//!
//! Each tensor `A[a, p, b]` becomes left-canonical if reshape to `(d_l*d_p, d_r)` is
//! column-orthonormal: `A^T A = I`.
//!
//! ## Right-canonical
//!
//! Each tensor `B[a, p, b]` becomes right-canonical if reshape to `(d_l, d_p*d_r)` is
//! row-orthonormal: `B B^T = I`.
//!
//! ## Mixed canonical (centred at site `s`)
//!
//! Sites `0..s` are left-canonical, sites `s+1..L` are right-canonical, and the residual
//! singular weight is carried into site `s`.

use crate::mps::Mps;
use crate::mps::tensor::MpsTensor;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// Canonical form tag for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Canonicalization {
    None,
    Left,
    Right,
    Mixed(usize),
}

/// Left-canonicalize every site of the MPS in place.
///
/// At the end, sites `0..L-1` are left-orthonormal and any residual weight is absorbed
/// into site `L-1`. The norm of the MPS is preserved.
pub fn left_canonicalize(mps: &mut Mps) -> TnResult<Canonicalization> {
    let n = mps.site_tensors.len();
    if n == 0 {
        return Err(TnError::EmptyInput);
    }
    for s in 0..n - 1 {
        let (dl, dp, dr) = mps.site_tensors[s].shape();
        let m = dl * dp;
        // SVD the matrix view of shape (m, dr): A = U * diag(s) * V^T
        let svd = svd_jacobi(&mps.site_tensors[s].data, m, dr)?;
        let k = svd.k;
        // New site tensor s := U reshaped to (dl, dp, k)
        let mut new_left = vec![0.0; dl * dp * k];
        for i in 0..m {
            for j in 0..k {
                new_left[i * k + j] = svd.u[i * k + j];
            }
        }
        mps.site_tensors[s] = MpsTensor::new(dl, dp, k, new_left)?;
        // Multiply diag(s)*V^T into next site: shape (k, dr) onto (dr, dp_next, dr_next)
        let (dr_old, dp_next, dr_next) = mps.site_tensors[s + 1].shape();
        if dr_old != dr {
            return Err(TnError::DimensionMismatch { a: dr_old, b: dr });
        }
        let mut new_next = vec![0.0; k * dp_next * dr_next];
        let next_data = &mps.site_tensors[s + 1].data;
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
        mps.site_tensors[s + 1] = MpsTensor::new(k, dp_next, dr_next, new_next)?;
    }
    Ok(Canonicalization::Left)
}

/// Right-canonicalize every site of the MPS in place.
pub fn right_canonicalize(mps: &mut Mps) -> TnResult<Canonicalization> {
    let n = mps.site_tensors.len();
    if n == 0 {
        return Err(TnError::EmptyInput);
    }
    for s in (1..n).rev() {
        let (dl, dp, dr) = mps.site_tensors[s].shape();
        let n_cols = dp * dr;
        // SVD the matrix view of shape (dl, dp*dr): A = U * diag(s) * V^T
        let svd = svd_jacobi(&mps.site_tensors[s].data, dl, n_cols)?;
        let k = svd.k;
        // New site tensor s := V^T reshaped to (k, dp, dr)
        let mut new_right = vec![0.0; k * dp * dr];
        for i in 0..k {
            for j in 0..n_cols {
                new_right[i * n_cols + j] = svd.vt[i * n_cols + j];
            }
        }
        mps.site_tensors[s] = MpsTensor::new(k, dp, dr, new_right)?;
        // Multiply U * diag(s) into previous site
        let (dl_prev, dp_prev, dr_prev_old) = mps.site_tensors[s - 1].shape();
        if dr_prev_old != dl {
            return Err(TnError::DimensionMismatch {
                a: dr_prev_old,
                b: dl,
            });
        }
        let mut new_prev = vec![0.0; dl_prev * dp_prev * k];
        let prev_data = &mps.site_tensors[s - 1].data;
        for a in 0..dl_prev {
            for p in 0..dp_prev {
                for new_b in 0..k {
                    let mut acc = 0.0;
                    for old_b in 0..dl {
                        let pv = prev_data[(a * dp_prev + p) * dr_prev_old + old_b];
                        let sv = if new_b < svd.s.len() {
                            svd.s[new_b]
                        } else {
                            0.0
                        };
                        let uv = svd.u[old_b * k + new_b];
                        acc += pv * uv * sv;
                    }
                    new_prev[(a * dp_prev + p) * k + new_b] = acc;
                }
            }
        }
        mps.site_tensors[s - 1] = MpsTensor::new(dl_prev, dp_prev, k, new_prev)?;
    }
    Ok(Canonicalization::Right)
}

/// Mixed-canonicalize: left-canonical on the left of `center`, right-canonical on the
/// right of `center`.
pub fn mixed_canonicalize(mps: &mut Mps, center: usize) -> TnResult<Canonicalization> {
    if center >= mps.n_sites() {
        return Err(TnError::IndexOutOfBounds {
            index: center,
            len: mps.n_sites(),
        });
    }
    // Left sweep up to but not including `center`
    for s in 0..center {
        let (dl, dp, dr) = mps.site_tensors[s].shape();
        let m = dl * dp;
        let svd = svd_jacobi(&mps.site_tensors[s].data, m, dr)?;
        let k = svd.k;
        let mut new_left = vec![0.0; dl * dp * k];
        for i in 0..m {
            for j in 0..k {
                new_left[i * k + j] = svd.u[i * k + j];
            }
        }
        mps.site_tensors[s] = MpsTensor::new(dl, dp, k, new_left)?;
        let (_, dp_next, dr_next) = mps.site_tensors[s + 1].shape();
        let mut new_next = vec![0.0; k * dp_next * dr_next];
        let next_data = &mps.site_tensors[s + 1].data;
        for new_a in 0..k {
            for p in 0..dp_next {
                for c in 0..dr_next {
                    let mut acc = 0.0;
                    for old_a in 0..dr {
                        let sv = svd.s[new_a];
                        let vtv = svd.vt[new_a * dr + old_a];
                        let nv = next_data[(old_a * dp_next + p) * dr_next + c];
                        acc += sv * vtv * nv;
                    }
                    new_next[(new_a * dp_next + p) * dr_next + c] = acc;
                }
            }
        }
        mps.site_tensors[s + 1] = MpsTensor::new(k, dp_next, dr_next, new_next)?;
    }
    // Right sweep down to but not including `center`
    let n = mps.n_sites();
    for s in (center + 1..n).rev() {
        let (dl, dp, dr) = mps.site_tensors[s].shape();
        let n_cols = dp * dr;
        let svd = svd_jacobi(&mps.site_tensors[s].data, dl, n_cols)?;
        let k = svd.k;
        let mut new_right = vec![0.0; k * dp * dr];
        for i in 0..k {
            for j in 0..n_cols {
                new_right[i * n_cols + j] = svd.vt[i * n_cols + j];
            }
        }
        mps.site_tensors[s] = MpsTensor::new(k, dp, dr, new_right)?;
        let (dl_prev, dp_prev, dr_prev_old) = mps.site_tensors[s - 1].shape();
        if dr_prev_old != dl {
            return Err(TnError::DimensionMismatch {
                a: dr_prev_old,
                b: dl,
            });
        }
        let mut new_prev = vec![0.0; dl_prev * dp_prev * k];
        let prev_data = &mps.site_tensors[s - 1].data;
        for a in 0..dl_prev {
            for p in 0..dp_prev {
                for new_b in 0..k {
                    let mut acc = 0.0;
                    let sv = svd.s[new_b];
                    for old_b in 0..dl {
                        let pv = prev_data[(a * dp_prev + p) * dr_prev_old + old_b];
                        let uv = svd.u[old_b * k + new_b];
                        acc += pv * uv * sv;
                    }
                    new_prev[(a * dp_prev + p) * k + new_b] = acc;
                }
            }
        }
        mps.site_tensors[s - 1] = MpsTensor::new(dl_prev, dp_prev, k, new_prev)?;
    }
    Ok(Canonicalization::Mixed(center))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn check_left_canonical(mps: &Mps, last_excluded: bool) -> bool {
        // For each non-final site, A^T A = I_(dr)
        let n = mps.n_sites();
        let stop = if last_excluded { n - 1 } else { n };
        for s in 0..stop {
            let t = &mps.site_tensors[s];
            let m = t.d_l * t.d_p;
            let dr = t.d_r;
            for i in 0..dr {
                for j in 0..dr {
                    let mut acc = 0.0;
                    for r in 0..m {
                        acc += t.data[r * dr + i] * t.data[r * dr + j];
                    }
                    let target = if i == j { 1.0 } else { 0.0 };
                    if (acc - target).abs() > 1e-8 {
                        return false;
                    }
                }
            }
        }
        true
    }

    #[test]
    fn left_canonical_random() {
        let mut rng = LcgRng::new(7);
        let mut mps = Mps::random_mps(4, 2, 3, &mut rng).expect("ok");
        left_canonicalize(&mut mps).expect("ok");
        assert!(check_left_canonical(&mps, true));
    }

    #[test]
    fn right_canonical_random() {
        let mut rng = LcgRng::new(11);
        let mut mps = Mps::random_mps(4, 2, 3, &mut rng).expect("ok");
        right_canonicalize(&mut mps).expect("ok");
        // For each non-first site, M reshape to (d_l, d_p*d_r) should be row-orthonormal
        for s in 1..mps.n_sites() {
            let t = &mps.site_tensors[s];
            let dl = t.d_l;
            let cols = t.d_p * t.d_r;
            for i in 0..dl {
                for j in 0..dl {
                    let mut acc = 0.0;
                    for c in 0..cols {
                        acc += t.data[i * cols + c] * t.data[j * cols + c];
                    }
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert!((acc - target).abs() < 1e-8);
                }
            }
        }
    }
}
