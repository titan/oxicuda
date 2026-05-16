//! Higher-Order Singular Value Decomposition (HOSVD) of a 3-mode tensor.
//!
//! Given `T` of shape `(d0, d1, d2)` we compute the Tucker decomposition
//! `T ≈ S ×_0 U0 ×_1 U1 ×_2 U2`
//! where `U_k` is the left singular factor of the mode-k unfolding of `T`
//! truncated to `r_k` columns and `S` is the resulting core tensor of shape
//! `(r_0, r_1, r_2)`.

use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// Output of the Tucker decomposition.
#[derive(Debug, Clone)]
pub struct TuckerResult {
    pub core: Vec<f64>,
    pub dims: (usize, usize, usize),
    pub ranks: (usize, usize, usize),
    pub u0: Vec<f64>, // (d0, r0)
    pub u1: Vec<f64>, // (d1, r1)
    pub u2: Vec<f64>, // (d2, r2)
}

/// Compute HOSVD of a 3-mode tensor, returning a [`TuckerResult`].
pub fn hosvd(
    t: &[f64],
    d0: usize,
    d1: usize,
    d2: usize,
    r0: usize,
    r1: usize,
    r2: usize,
) -> TnResult<TuckerResult> {
    if d0 == 0 || d1 == 0 || d2 == 0 {
        return Err(TnError::EmptyInput);
    }
    let total = d0 * d1 * d2;
    if t.len() != total {
        return Err(TnError::ShapeMismatch {
            expected: vec![d0, d1, d2],
            got: vec![t.len()],
        });
    }
    if r0 == 0 || r1 == 0 || r2 == 0 || r0 > d0 || r1 > d1 || r2 > d2 {
        return Err(TnError::InvalidRank(r0.max(r1).max(r2)));
    }

    let u0 = mode_unfold_then_svd_left(t, d0, d1, d2, 0, r0)?;
    let u1 = mode_unfold_then_svd_left(t, d0, d1, d2, 1, r1)?;
    let u2 = mode_unfold_then_svd_left(t, d0, d1, d2, 2, r2)?;
    // Core = T ×_0 U0^T ×_1 U1^T ×_2 U2^T
    let core = mode_apply_left_transpose(t, &u0, &u1, &u2, d0, d1, d2, r0, r1, r2);
    Ok(TuckerResult {
        core,
        dims: (d0, d1, d2),
        ranks: (r0, r1, r2),
        u0,
        u1,
        u2,
    })
}

/// Unfold along `mode`, truncate the leading singular vectors to `r` columns.
pub fn mode_unfold_then_svd_left(
    t: &[f64],
    d0: usize,
    d1: usize,
    d2: usize,
    mode: usize,
    r: usize,
) -> TnResult<Vec<f64>> {
    let (m, n, unfolded) = mode_unfold(t, d0, d1, d2, mode);
    let svd = svd_jacobi(&unfolded, m, n)?;
    if r > svd.k {
        return Err(TnError::InvalidRank(r));
    }
    let mut u = vec![0.0; m * r];
    for i in 0..m {
        for j in 0..r {
            u[i * r + j] = svd.u[i * svd.k + j];
        }
    }
    Ok(u)
}

/// Mode-unfold a 3-tensor and return `(rows, cols, matrix)`.
///
/// * mode 0: rows = d0, cols = d1*d2, layout `[i, j*d2 + k]`
/// * mode 1: rows = d1, cols = d0*d2, layout `[j, i*d2 + k]`
/// * mode 2: rows = d2, cols = d0*d1, layout `[k, i*d1 + j]`
pub fn mode_unfold(
    t: &[f64],
    d0: usize,
    d1: usize,
    d2: usize,
    mode: usize,
) -> (usize, usize, Vec<f64>) {
    match mode {
        0 => {
            let m = d0;
            let n = d1 * d2;
            let mut out = vec![0.0; m * n];
            for i in 0..d0 {
                for j in 0..d1 {
                    for k in 0..d2 {
                        out[i * n + j * d2 + k] = t[(i * d1 + j) * d2 + k];
                    }
                }
            }
            (m, n, out)
        }
        1 => {
            let m = d1;
            let n = d0 * d2;
            let mut out = vec![0.0; m * n];
            for i in 0..d0 {
                for j in 0..d1 {
                    for k in 0..d2 {
                        out[j * n + i * d2 + k] = t[(i * d1 + j) * d2 + k];
                    }
                }
            }
            (m, n, out)
        }
        _ => {
            // mode 2 (default)
            let m = d2;
            let n = d0 * d1;
            let mut out = vec![0.0; m * n];
            for i in 0..d0 {
                for j in 0..d1 {
                    for k in 0..d2 {
                        out[k * n + i * d1 + j] = t[(i * d1 + j) * d2 + k];
                    }
                }
            }
            (m, n, out)
        }
    }
}

/// Apply `T ×_0 U0^T ×_1 U1^T ×_2 U2^T` to obtain the core tensor of shape
/// `(r0, r1, r2)` row-major.
#[allow(clippy::too_many_arguments)]
pub fn mode_apply_left_transpose(
    t: &[f64],
    u0: &[f64],
    u1: &[f64],
    u2: &[f64],
    d0: usize,
    d1: usize,
    d2: usize,
    r0: usize,
    r1: usize,
    r2: usize,
) -> Vec<f64> {
    // Step 1: T_1[a, j, k] = sum_i U0[i, a] * T[i, j, k]
    let mut t1 = vec![0.0; r0 * d1 * d2];
    for a in 0..r0 {
        for j in 0..d1 {
            for k in 0..d2 {
                let mut acc = 0.0;
                for i in 0..d0 {
                    acc += u0[i * r0 + a] * t[(i * d1 + j) * d2 + k];
                }
                t1[(a * d1 + j) * d2 + k] = acc;
            }
        }
    }
    // Step 2: T_2[a, b, k] = sum_j U1[j, b] * T_1[a, j, k]
    let mut t2 = vec![0.0; r0 * r1 * d2];
    for a in 0..r0 {
        for b in 0..r1 {
            for k in 0..d2 {
                let mut acc = 0.0;
                for j in 0..d1 {
                    acc += u1[j * r1 + b] * t1[(a * d1 + j) * d2 + k];
                }
                t2[(a * r1 + b) * d2 + k] = acc;
            }
        }
    }
    // Step 3: S[a, b, c] = sum_k U2[k, c] * T_2[a, b, k]
    let mut s = vec![0.0; r0 * r1 * r2];
    for a in 0..r0 {
        for b in 0..r1 {
            for c in 0..r2 {
                let mut acc = 0.0;
                for k in 0..d2 {
                    acc += u2[k * r2 + c] * t2[(a * r1 + b) * d2 + k];
                }
                s[(a * r1 + b) * r2 + c] = acc;
            }
        }
    }
    s
}

/// Reconstruct a 3-tensor from a Tucker decomposition.
pub fn tucker_reconstruct(res: &TuckerResult) -> Vec<f64> {
    let (d0, d1, d2) = res.dims;
    let (r0, r1, r2) = res.ranks;
    // Step 1: T_1[i, b, c] = sum_a U0[i, a] * S[a, b, c]
    let mut t1 = vec![0.0; d0 * r1 * r2];
    for i in 0..d0 {
        for b in 0..r1 {
            for c in 0..r2 {
                let mut acc = 0.0;
                for a in 0..r0 {
                    acc += res.u0[i * r0 + a] * res.core[(a * r1 + b) * r2 + c];
                }
                t1[(i * r1 + b) * r2 + c] = acc;
            }
        }
    }
    // Step 2: T_2[i, j, c] = sum_b U1[j, b] * T_1[i, b, c]
    let mut t2 = vec![0.0; d0 * d1 * r2];
    for i in 0..d0 {
        for j in 0..d1 {
            for c in 0..r2 {
                let mut acc = 0.0;
                for b in 0..r1 {
                    acc += res.u1[j * r1 + b] * t1[(i * r1 + b) * r2 + c];
                }
                t2[(i * d1 + j) * r2 + c] = acc;
            }
        }
    }
    // Step 3: T[i, j, k] = sum_c U2[k, c] * T_2[i, j, c]
    let mut out = vec![0.0; d0 * d1 * d2];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                let mut acc = 0.0;
                for c in 0..r2 {
                    acc += res.u2[k * r2 + c] * t2[(i * d1 + j) * r2 + c];
                }
                out[(i * d1 + j) * d2 + k] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn fro_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn hosvd_full_rank_roundtrip() {
        let mut rng = LcgRng::new(7);
        let d0 = 3;
        let d1 = 2;
        let d2 = 4;
        let data: Vec<f64> = (0..d0 * d1 * d2).map(|_| rng.next_normal()).collect();
        let res = hosvd(&data, d0, d1, d2, d0, d1, d2).expect("ok");
        let rec = tucker_reconstruct(&res);
        assert!(fro_diff(&data, &rec) < 1e-8);
    }

    #[test]
    fn hosvd_rank1_input() {
        // Rank-1 tensor: outer product of three vectors
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        let c = [6.0, 7.0, 8.0, 9.0];
        let d0 = 3;
        let d1 = 2;
        let d2 = 4;
        let mut data = vec![0.0; d0 * d1 * d2];
        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    data[(i * d1 + j) * d2 + k] = a[i] * b[j] * c[k];
                }
            }
        }
        let res = hosvd(&data, d0, d1, d2, 1, 1, 1).expect("ok");
        let rec = tucker_reconstruct(&res);
        assert!(fro_diff(&data, &rec) < 1e-8);
    }
}
