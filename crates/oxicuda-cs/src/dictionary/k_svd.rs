//! K-SVD dictionary learning (Aharon, Elad, Bruckstein 2006).
//!
//! Alternates between:
//!  - Sparse coding (OMP per data column)
//!  - Per-atom SVD update of the dictionary

use crate::dictionary::DictionaryResult;
use crate::error::{CsError, CsResult};
use crate::greedy::omp::omp;
use crate::handle::LcgRng;
use crate::linalg::jacobi_svd::jacobi_svd_thin;

/// K-SVD with `n_atoms` columns, target sparsity `sparsity` per signal.
///
/// `signals`: `d × n_samples` row-major data matrix.
pub fn k_svd(
    signals: &[f64],
    d: usize,
    n_samples: usize,
    n_atoms: usize,
    sparsity: usize,
    max_iter: usize,
    tol: f64,
    rng: &mut LcgRng,
) -> CsResult<DictionaryResult> {
    if signals.len() != d * n_samples {
        return Err(CsError::ShapeMismatch {
            expected: vec![d, n_samples],
            got: vec![signals.len()],
        });
    }
    if n_atoms == 0 || n_atoms > n_samples {
        return Err(CsError::InvalidRank(n_atoms));
    }
    if sparsity == 0 || sparsity > n_atoms {
        return Err(CsError::InvalidSparsity(sparsity));
    }
    // Initialise dictionary by picking `n_atoms` random columns of `signals`, normalised.
    let mut dict = vec![0.0_f64; d * n_atoms];
    let mut chosen = vec![false; n_samples];
    for k in 0..n_atoms {
        let mut idx = rng.next_usize(n_samples);
        while chosen[idx] {
            idx = (idx + 1) % n_samples;
        }
        chosen[idx] = true;
        let mut nrm = 0.0_f64;
        for i in 0..d {
            let v = signals[i * n_samples + idx];
            dict[i * n_atoms + k] = v;
            nrm += v * v;
        }
        let nrm = nrm.sqrt().max(1.0e-300);
        for i in 0..d {
            dict[i * n_atoms + k] /= nrm;
        }
    }
    let mut codes = vec![0.0_f64; n_atoms * n_samples];
    let mut iter = 0usize;
    let mut last_err = f64::INFINITY;
    for _ in 0..max_iter {
        // Sparse coding via OMP per column of signals.
        for j in 0..n_samples {
            let mut y = vec![0.0_f64; d];
            for i in 0..d {
                y[i] = signals[i * n_samples + j];
            }
            let r = omp(&dict, d, n_atoms, &y, sparsity, 1.0e-12)?;
            for kk in 0..n_atoms {
                codes[kk * n_samples + j] = r.x[kk];
            }
        }
        // Dictionary update: for each atom k, compute the residual E_k restricted to columns
        // using k, then top-1 SVD updates dict[:,k] and codes[k, those_columns].
        for kk in 0..n_atoms {
            let mut omega: Vec<usize> = Vec::new();
            for j in 0..n_samples {
                if codes[kk * n_samples + j].abs() > 1.0e-12 {
                    omega.push(j);
                }
            }
            if omega.is_empty() {
                continue;
            }
            // Build residual matrix E_k (d × |omega|) in row-major.
            let nw = omega.len();
            let mut e_k = vec![0.0_f64; d * nw];
            for (col_idx, &j) in omega.iter().enumerate() {
                for i in 0..d {
                    let mut yij = signals[i * n_samples + j];
                    // Subtract contributions of other atoms.
                    for kp in 0..n_atoms {
                        if kp == kk {
                            continue;
                        }
                        yij -= dict[i * n_atoms + kp] * codes[kp * n_samples + j];
                    }
                    e_k[i * nw + col_idx] = yij;
                }
            }
            // Top-1 SVD on e_k (d × nw).
            if d >= nw {
                let (u, s, v) = jacobi_svd_thin(&e_k, d, nw)?;
                // Update dict[:,kk] = u[:,0], codes[kk, omega] = s[0] * v[:,0].
                for i in 0..d {
                    dict[i * n_atoms + kk] = u[i * nw];
                }
                for (col_idx, &j) in omega.iter().enumerate() {
                    codes[kk * n_samples + j] = s[0] * v[col_idx * nw];
                }
            } else {
                // d < nw: factor e_k^T.
                let mut t = vec![0.0_f64; nw * d];
                for i in 0..d {
                    for cj in 0..nw {
                        t[cj * d + i] = e_k[i * nw + cj];
                    }
                }
                let (u, s, v) = jacobi_svd_thin(&t, nw, d)?;
                // e_k = (u s v^T)^T = v s u^T => dict[:,kk] = v[:,0], codes[kk,omega] = s[0] * u[:,0]
                let dmin = d.min(nw);
                for i in 0..d {
                    dict[i * n_atoms + kk] = v[i * dmin];
                }
                for (col_idx, &j) in omega.iter().enumerate() {
                    codes[kk * n_samples + j] = s[0] * u[col_idx * dmin];
                }
            }
        }
        // Reconstruction error.
        let mut err = 0.0_f64;
        for i in 0..d {
            for j in 0..n_samples {
                let mut acc = 0.0_f64;
                for kk in 0..n_atoms {
                    acc += dict[i * n_atoms + kk] * codes[kk * n_samples + j];
                }
                let dij = signals[i * n_samples + j] - acc;
                err += dij * dij;
            }
        }
        err = err.sqrt();
        iter += 1;
        if (last_err - err).abs() < tol {
            break;
        }
        last_err = err;
    }
    Ok(DictionaryResult {
        dict,
        codes,
        iterations: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_svd_runs() {
        let mut rng = LcgRng::new(42);
        let d = 6;
        let n_samples = 8;
        let n_atoms = 4;
        // Random data; we just need this to not crash.
        let signals: Vec<f64> = (0..(d * n_samples))
            .map(|i| (i as f64 % 5.0) - 2.0)
            .collect();
        let r = k_svd(&signals, d, n_samples, n_atoms, 2, 5, 1.0e-6, &mut rng).expect("ok");
        assert_eq!(r.dict.len(), d * n_atoms);
        assert_eq!(r.codes.len(), n_atoms * n_samples);
    }
}
