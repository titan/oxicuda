//! Viterbi decoding for linear-chain CRFs (score-space).

use super::linear_chain_crf::LinearChainCrf;
use crate::error::{SeqError, SeqResult};

/// Viterbi decoding for a linear-chain CRF.  Returns the label sequence that
/// maximises the un-normalised CRF score (equivalent to argmax of the conditional
/// posterior because the partition function is constant in `y`).
pub fn viterbi_decode(crf: &LinearChainCrf, x: &[f64]) -> SeqResult<Vec<usize>> {
    let n = crf.n_labels;
    let k = crf.n_features;
    if x.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    if x.len() % k != 0 {
        return Err(SeqError::DimensionMismatch { a: x.len(), b: k });
    }
    let t_max = x.len() / k;

    let mut delta = vec![f64::NEG_INFINITY; t_max * n];
    let mut psi = vec![0usize; t_max * n];

    // t = 0
    for j in 0..n {
        delta[j] = crf.emit_score(j, &x[..k])?;
    }
    for t in 1..t_max {
        for j in 0..n {
            let emit = crf.emit_score(j, &x[t * k..(t + 1) * k])?;
            let mut best = f64::NEG_INFINITY;
            let mut argmax = 0usize;
            for i in 0..n {
                let v = delta[(t - 1) * n + i] + crf.transitions[i * n + j];
                if v > best {
                    best = v;
                    argmax = i;
                }
            }
            delta[t * n + j] = best + emit;
            psi[t * n + j] = argmax;
        }
    }

    let mut best = f64::NEG_INFINITY;
    let mut last = 0usize;
    for j in 0..n {
        let v = delta[(t_max - 1) * n + j];
        if v > best {
            best = v;
            last = j;
        }
    }

    let mut path = vec![0usize; t_max];
    path[t_max - 1] = last;
    for t in (1..t_max).rev() {
        path[t - 1] = psi[t * n + path[t]];
    }

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_trivial() {
        let mut crf = LinearChainCrf::zeros(2, 2).expect("ok");
        // Label 0 weighs feature 0, label 1 weighs feature 1.
        crf.emissions = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![5.0, 0.0, 0.0, 5.0, 5.0, 0.0];
        let y = viterbi_decode(&crf, &x).expect("ok");
        assert_eq!(y, vec![0, 1, 0]);
    }
}
