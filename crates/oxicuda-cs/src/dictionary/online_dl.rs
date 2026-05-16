//! Online dictionary learning (Mairal, Bach, Ponce, Sapiro 2010).
//!
//! Processes signals one at a time, maintaining sufficient statistics `A = Σ α_t α_t^T` and
//! `B = Σ x_t α_t^T`. At each step a block-coordinate descent updates the dictionary.

use crate::dictionary::DictionaryResult;
use crate::error::{CsError, CsResult};
use crate::greedy::omp::omp;
use crate::handle::LcgRng;

/// Online dictionary learning over a single epoch of the given signals.
pub fn online_dl(
    signals: &[f64],
    d: usize,
    n_samples: usize,
    n_atoms: usize,
    sparsity: usize,
    n_epochs: usize,
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
    if n_epochs == 0 {
        return Err(CsError::InvalidParameter("n_epochs must be > 0".into()));
    }
    // Initialise dict by sampling unique columns.
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
    let mut a_stat = vec![0.0_f64; n_atoms * n_atoms];
    let mut b_stat = vec![0.0_f64; d * n_atoms];
    let mut codes = vec![0.0_f64; n_atoms * n_samples];
    let mut iter = 0usize;
    for _ in 0..n_epochs {
        // Random permutation of indices via Fisher-Yates.
        let mut indices: Vec<usize> = (0..n_samples).collect();
        for i in 0..n_samples {
            let j = i + rng.next_usize(n_samples - i);
            indices.swap(i, j);
        }
        for &t in &indices {
            // Sparse coding on current dict.
            let mut y = vec![0.0_f64; d];
            for i in 0..d {
                y[i] = signals[i * n_samples + t];
            }
            let r = omp(&dict, d, n_atoms, &y, sparsity, 1.0e-12)?;
            let alpha = r.x;
            for kk in 0..n_atoms {
                codes[kk * n_samples + t] = alpha[kk];
            }
            // Update A and B.
            for a in 0..n_atoms {
                for b in 0..n_atoms {
                    a_stat[a * n_atoms + b] += alpha[a] * alpha[b];
                }
            }
            for i in 0..d {
                for a in 0..n_atoms {
                    b_stat[i * n_atoms + a] += y[i] * alpha[a];
                }
            }
            // Block-coordinate descent for each atom.
            for j in 0..n_atoms {
                let ajj = a_stat[j * n_atoms + j];
                if ajj.abs() < 1.0e-12 {
                    continue;
                }
                let mut u = vec![0.0_f64; d];
                for i in 0..d {
                    let mut s = b_stat[i * n_atoms + j];
                    for a in 0..n_atoms {
                        s -= a_stat[a * n_atoms + j] * dict[i * n_atoms + a];
                    }
                    s += ajj * dict[i * n_atoms + j];
                    u[i] = s / ajj;
                }
                let mut nrm = 0.0_f64;
                for &v in &u {
                    nrm += v * v;
                }
                let nrm = nrm.sqrt().max(1.0e-300);
                let scale = if nrm > 1.0 { 1.0 / nrm } else { 1.0 };
                for i in 0..d {
                    dict[i * n_atoms + j] = u[i] * scale;
                }
            }
            iter += 1;
        }
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
    fn online_dl_runs() {
        let mut rng = LcgRng::new(31);
        let d = 5;
        let n_samples = 8;
        let n_atoms = 3;
        let signals: Vec<f64> = (0..(d * n_samples)).map(|i| (i as f64) * 0.05).collect();
        let r = online_dl(&signals, d, n_samples, n_atoms, 2, 2, &mut rng).expect("ok");
        assert_eq!(r.dict.len(), d * n_atoms);
        assert_eq!(r.codes.len(), n_atoms * n_samples);
    }
}
