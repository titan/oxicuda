//! TT-SVD algorithm (Oseledets, 2011) for compressing a dense tensor into TT format.

use crate::svd::svd_jacobi;
use crate::tt::tt::{TtCore, TtTensor};
use crate::{TnError, TnResult};

/// TT-SVD: convert a flat C-order tensor of shape `dims` into a TT-tensor with maximal
/// bond dimension `r_max` and truncation tolerance `tol`.
pub fn tt_svd(data: &[f64], dims: &[usize], r_max: usize, tol: f64) -> TnResult<TtTensor> {
    if dims.is_empty() {
        return Err(TnError::EmptyInput);
    }
    let total: usize = dims.iter().product();
    if data.len() != total {
        return Err(TnError::ShapeMismatch {
            expected: vec![total],
            got: vec![data.len()],
        });
    }
    let d = dims.len();
    let mut cores: Vec<TtCore> = Vec::with_capacity(d);
    let mut current = data.to_vec();
    let mut r_k = 1usize;
    let mut remaining_size = total;
    for &n_k in dims.iter().take(d - 1) {
        let rows = r_k * n_k;
        let cols = remaining_size / n_k;
        // SVD of the (rows, cols) reshape
        let svd = svd_jacobi(&current, rows, cols)?;
        // Determine truncated rank
        let s_max = svd.s.first().copied().unwrap_or(0.0);
        let abs_tol = tol * s_max.max(1.0);
        let mut keep = 0usize;
        for &v in &svd.s {
            if keep >= r_max {
                break;
            }
            if v < abs_tol {
                break;
            }
            keep += 1;
        }
        keep = keep.max(1);
        // Build core G_k of shape (r_k, n_k, keep)
        let mut core_data = vec![0.0; r_k * n_k * keep];
        for i in 0..rows {
            for j in 0..keep {
                core_data[i * keep + j] = svd.u[i * svd.k + j];
            }
        }
        cores.push(TtCore::new(r_k, n_k, keep, core_data)?);
        // current := diag(s)[0..keep] * V^T[0..keep, :] of shape (keep, cols)
        let mut new_current = vec![0.0; keep * cols];
        for i in 0..keep {
            let sv = svd.s[i];
            for j in 0..cols {
                new_current[i * cols + j] = sv * svd.vt[i * cols + j];
            }
        }
        current = new_current;
        r_k = keep;
        remaining_size = cols;
    }
    // Final core: shape (r_k, n_{d-1}, 1)
    let n_last = dims[d - 1];
    if current.len() != r_k * n_last {
        return Err(TnError::ShapeMismatch {
            expected: vec![r_k * n_last],
            got: vec![current.len()],
        });
    }
    cores.push(TtCore::new(r_k, n_last, 1, current)?);
    TtTensor::new(cores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn fro(a: &[f64]) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn tt_svd_roundtrip_small() {
        let mut rng = LcgRng::new(13);
        let dims = vec![3, 4, 2];
        let total: usize = dims.iter().product();
        let data: Vec<f64> = (0..total).map(|_| rng.next_normal()).collect();
        let tt = tt_svd(&data, &dims, 20, 1e-14).expect("ok");
        let rec = tt.reconstruct().expect("ok");
        let diff: Vec<f64> = data.iter().zip(&rec).map(|(a, b)| a - b).collect();
        assert!(fro(&diff) < 1e-8, "fro diff = {}", fro(&diff));
    }

    #[test]
    fn tt_svd_rank1_input() {
        // Rank-1 outer product: A[i, j] = u[i] v[j]
        let u = [1.0, 2.0, 3.0];
        let v = [4.0, 5.0];
        let mut data = vec![0.0; 6];
        for i in 0..3 {
            for j in 0..2 {
                data[i * 2 + j] = u[i] * v[j];
            }
        }
        let tt = tt_svd(&data, &[3, 2], 4, 1e-14).expect("ok");
        // Bond should be 1
        assert_eq!(tt.cores[0].r_r, 1);
        let rec = tt.reconstruct().expect("ok");
        let diff: Vec<f64> = data.iter().zip(&rec).map(|(a, b)| a - b).collect();
        assert!(fro(&diff) < 1e-10);
    }
}
