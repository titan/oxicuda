//! CP decomposition for 3-mode tensors via Alternating Least Squares (ALS).
//!
//! Approximate `T[i, j, k] ≈ sum_r λ_r A[i, r] B[j, r] C[k, r]`. We optimise `A`, `B`,
//! `C` with the normal equations using a Khatri-Rao formulation.

use crate::handle::LcgRng;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// CP decomposition result for a 3-mode tensor.
#[derive(Debug, Clone)]
pub struct CpResult {
    pub rank: usize,
    pub a: Vec<f64>, // (d0, rank)
    pub b: Vec<f64>, // (d1, rank)
    pub c: Vec<f64>, // (d2, rank)
    pub lambda: Vec<f64>,
    pub residual: f64,
    pub iter: usize,
}

/// Run ALS for `max_iter` iterations.
pub fn cp_als(
    t: &[f64],
    d0: usize,
    d1: usize,
    d2: usize,
    rank: usize,
    max_iter: usize,
    tol: f64,
    rng: &mut LcgRng,
) -> TnResult<CpResult> {
    if d0 == 0 || d1 == 0 || d2 == 0 || rank == 0 {
        return Err(TnError::EmptyInput);
    }
    if t.len() != d0 * d1 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d0, d1, d2],
            got: vec![t.len()],
        });
    }
    // Work with unnormalised factors during ALS; split column norms into `lambda` only
    // when reporting the final result.
    let mut a: Vec<f64> = (0..d0 * rank).map(|_| rng.next_normal()).collect();
    let mut b: Vec<f64> = (0..d1 * rank).map(|_| rng.next_normal()).collect();
    let mut c: Vec<f64> = (0..d2 * rank).map(|_| rng.next_normal()).collect();
    let mut prev_residual = f64::INFINITY;
    let mut iter_done = 0usize;
    let unfold0 = mode0_unfold(t, d0, d1, d2);
    let unfold1 = mode1_unfold(t, d0, d1, d2);
    let unfold2 = mode2_unfold(t, d0, d1, d2);
    for it in 0..max_iter {
        iter_done = it + 1;
        // Update A: A := T_(0) * (B ⊙ C) * pinv((B^T B) ⊙ (C^T C))
        let kr_bc = khatri_rao(&b, &c, d1, d2, rank);
        let gram = hadamard_gram(&b, &c, d1, d2, rank);
        a = ls_solve(&unfold0, d0, d1 * d2, &kr_bc, d1 * d2, rank, &gram, rank)?;
        // Update B
        let kr_ac = khatri_rao(&a, &c, d0, d2, rank);
        let gram = hadamard_gram(&a, &c, d0, d2, rank);
        b = ls_solve(&unfold1, d1, d0 * d2, &kr_ac, d0 * d2, rank, &gram, rank)?;
        // Update C
        let kr_ab = khatri_rao(&a, &b, d0, d1, rank);
        let gram = hadamard_gram(&a, &b, d0, d1, rank);
        c = ls_solve(&unfold2, d2, d0 * d1, &kr_ab, d0 * d1, rank, &gram, rank)?;
        let ones_lambda = vec![1.0; rank];
        let residual = cp_residual(t, &a, &b, &c, &ones_lambda, d0, d1, d2, rank);
        if (prev_residual - residual).abs() < tol && it > 0 {
            break;
        }
        prev_residual = residual;
    }
    // Final split into λ + normalised factors.
    let mut lambda = vec![1.0; rank];
    normalize_columns_with_lambda(&mut a, d0, rank, &mut lambda);
    normalize_columns_with_lambda(&mut b, d1, rank, &mut lambda);
    normalize_columns_with_lambda(&mut c, d2, rank, &mut lambda);
    let residual = cp_residual(t, &a, &b, &c, &lambda, d0, d1, d2, rank);
    Ok(CpResult {
        rank,
        a,
        b,
        c,
        lambda,
        residual,
        iter: iter_done,
    })
}

fn mode0_unfold(t: &[f64], d0: usize, d1: usize, d2: usize) -> Vec<f64> {
    // (d0, d1*d2)
    let n = d1 * d2;
    let mut out = vec![0.0; d0 * n];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                out[i * n + j * d2 + k] = t[(i * d1 + j) * d2 + k];
            }
        }
    }
    out
}

fn mode1_unfold(t: &[f64], d0: usize, d1: usize, d2: usize) -> Vec<f64> {
    // (d1, d0*d2)
    let n = d0 * d2;
    let mut out = vec![0.0; d1 * n];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                out[j * n + i * d2 + k] = t[(i * d1 + j) * d2 + k];
            }
        }
    }
    out
}

fn mode2_unfold(t: &[f64], d0: usize, d1: usize, d2: usize) -> Vec<f64> {
    // (d2, d0*d1)
    let n = d0 * d1;
    let mut out = vec![0.0; d2 * n];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                out[k * n + i * d1 + j] = t[(i * d1 + j) * d2 + k];
            }
        }
    }
    out
}

/// Khatri-Rao product of `A` (d_a × r) and `B` (d_b × r) → (d_a * d_b, r)
/// with rows ordered (a, b) such that out[a*d_b + b, r] = A[a, r] * B[b, r].
fn khatri_rao(a: &[f64], b: &[f64], d_a: usize, d_b: usize, r: usize) -> Vec<f64> {
    let mut out = vec![0.0; d_a * d_b * r];
    for i in 0..d_a {
        for j in 0..d_b {
            for k in 0..r {
                out[(i * d_b + j) * r + k] = a[i * r + k] * b[j * r + k];
            }
        }
    }
    out
}

/// Hadamard of two Gram matrices: returns `(A^T A) * (B^T B)` element-wise (r × r).
fn hadamard_gram(a: &[f64], b: &[f64], d_a: usize, d_b: usize, r: usize) -> Vec<f64> {
    let mut ga = vec![0.0; r * r];
    let mut gb = vec![0.0; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut sa = 0.0;
            for k in 0..d_a {
                sa += a[k * r + i] * a[k * r + j];
            }
            ga[i * r + j] = sa;
            let mut sb = 0.0;
            for k in 0..d_b {
                sb += b[k * r + i] * b[k * r + j];
            }
            gb[i * r + j] = sb;
        }
    }
    let mut out = vec![0.0; r * r];
    for i in 0..r {
        for j in 0..r {
            out[i * r + j] = ga[i * r + j] * gb[i * r + j];
        }
    }
    out
}

/// Solve the least-squares step `factor = unfold * KR * inv(gram + eps I)`.
#[allow(clippy::too_many_arguments)]
fn ls_solve(
    unfold: &[f64],
    d_out: usize,
    n_cols: usize,
    kr: &[f64],
    kr_rows: usize,
    kr_cols: usize,
    gram: &[f64],
    r: usize,
) -> TnResult<Vec<f64>> {
    if n_cols != kr_rows {
        return Err(TnError::DimensionMismatch {
            a: n_cols,
            b: kr_rows,
        });
    }
    if r != kr_cols {
        return Err(TnError::DimensionMismatch { a: r, b: kr_cols });
    }
    // Step 1: M = unfold * kr → shape (d_out, r)
    let mut m = vec![0.0; d_out * r];
    for i in 0..d_out {
        for c in 0..r {
            let mut acc = 0.0;
            for j in 0..n_cols {
                acc += unfold[i * n_cols + j] * kr[j * r + c];
            }
            m[i * r + c] = acc;
        }
    }
    // Step 2: Solve (gram + eps I) X^T = M^T column-by-column via pseudoinverse (SVD).
    let mut reg = gram.to_vec();
    let eps = 1e-12;
    for i in 0..r {
        reg[i * r + i] += eps;
    }
    let svd = svd_jacobi(&reg, r, r)?;
    // Pseudoinverse: V * diag(1/s) * U^T
    let mut pinv = vec![0.0; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut acc = 0.0;
            for s in 0..svd.k {
                if svd.s[s] > 1e-15 {
                    acc += svd.vt[s * r + i] * svd.u[j * r + s] / svd.s[s];
                }
            }
            pinv[i * r + j] = acc;
        }
    }
    // factor = M * pinv → (d_out, r)
    let mut out = vec![0.0; d_out * r];
    for i in 0..d_out {
        for j in 0..r {
            let mut acc = 0.0;
            for c in 0..r {
                acc += m[i * r + c] * pinv[c * r + j];
            }
            out[i * r + j] = acc;
        }
    }
    Ok(out)
}

fn normalize_columns_with_lambda(mat: &mut [f64], rows: usize, cols: usize, lambda: &mut [f64]) {
    for c in 0..cols {
        let mut nrm2 = 0.0;
        for r in 0..rows {
            nrm2 += mat[r * cols + c] * mat[r * cols + c];
        }
        if nrm2 > 1e-300 {
            let nrm = nrm2.sqrt();
            for r in 0..rows {
                mat[r * cols + c] /= nrm;
            }
            lambda[c] *= nrm;
        }
    }
}

fn cp_residual(
    t: &[f64],
    a: &[f64],
    b: &[f64],
    c: &[f64],
    lambda: &[f64],
    d0: usize,
    d1: usize,
    d2: usize,
    rank: usize,
) -> f64 {
    let mut acc = 0.0;
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                let mut rec = 0.0;
                for r in 0..rank {
                    rec += lambda[r] * a[i * rank + r] * b[j * rank + r] * c[k * rank + r];
                }
                let diff = t[(i * d1 + j) * d2 + k] - rec;
                acc += diff * diff;
            }
        }
    }
    acc.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn cp_als_rank1_recovery() {
        let a = [1.0, 2.0, 3.0];
        let b = [0.5, 4.0];
        let c = [1.0, 0.5, 2.0];
        let d0 = 3;
        let d1 = 2;
        let d2 = 3;
        let mut data = vec![0.0; d0 * d1 * d2];
        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    data[(i * d1 + j) * d2 + k] = a[i] * b[j] * c[k];
                }
            }
        }
        let mut rng = LcgRng::new(11);
        let res = cp_als(&data, d0, d1, d2, 1, 100, 1e-10, &mut rng).expect("ok");
        assert!(res.residual < 1e-6, "residual = {}", res.residual);
    }
}
