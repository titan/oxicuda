//! Method of Optimal Directions (Engan, Aase, Husøy 1999).
//!
//! Alternates between OMP sparse coding and a closed-form least-squares dictionary update:
//!   D ← X Cᵀ (C Cᵀ)⁻¹

use crate::dictionary::DictionaryResult;
use crate::error::{CsError, CsResult};
use crate::greedy::omp::omp;
use crate::handle::LcgRng;
use crate::linalg::cholesky::{cholesky_factor, cholesky_solve};

/// Method of Optimal Directions.
pub fn mod_dl(
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
        // Sparse coding.
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
        // Dictionary update: D ← X C^T (C C^T)^{-1}.
        // C is n_atoms × n_samples row-major.
        // C C^T is n_atoms × n_atoms.
        let mut cct = vec![0.0_f64; n_atoms * n_atoms];
        for j in 0..n_samples {
            for a in 0..n_atoms {
                let caj = codes[a * n_samples + j];
                for b in 0..n_atoms {
                    cct[a * n_atoms + b] += caj * codes[b * n_samples + j];
                }
            }
        }
        // Regularise for stability.
        for a in 0..n_atoms {
            cct[a * n_atoms + a] += 1.0e-8;
        }
        let l = cholesky_factor(&cct, n_atoms)?;
        // Compute X C^T (d × n_atoms).
        let mut xct = vec![0.0_f64; d * n_atoms];
        for i in 0..d {
            for a in 0..n_atoms {
                let mut s = 0.0_f64;
                for j in 0..n_samples {
                    s += signals[i * n_samples + j] * codes[a * n_samples + j];
                }
                xct[i * n_atoms + a] = s;
            }
        }
        // For each row i of xct, solve (cct) z = xct_row, then row i of dict = z.
        for i in 0..d {
            let mut row = vec![0.0_f64; n_atoms];
            for a in 0..n_atoms {
                row[a] = xct[i * n_atoms + a];
            }
            let z = cholesky_solve(&l, n_atoms, &row)?;
            for a in 0..n_atoms {
                dict[i * n_atoms + a] = z[a];
            }
        }
        // Renormalise atoms.
        for a in 0..n_atoms {
            let mut nrm = 0.0_f64;
            for i in 0..d {
                let v = dict[i * n_atoms + a];
                nrm += v * v;
            }
            let nrm = nrm.sqrt().max(1.0e-300);
            for i in 0..d {
                dict[i * n_atoms + a] /= nrm;
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
    fn mod_runs() {
        let mut rng = LcgRng::new(7);
        let d = 5;
        let n_samples = 6;
        let n_atoms = 3;
        let signals: Vec<f64> = (0..(d * n_samples)).map(|i| (i as f64) * 0.1).collect();
        let r = mod_dl(&signals, d, n_samples, n_atoms, 2, 3, 1.0e-6, &mut rng).expect("ok");
        assert_eq!(r.dict.len(), d * n_atoms);
    }
}
