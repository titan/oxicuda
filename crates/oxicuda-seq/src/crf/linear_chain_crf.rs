//! Linear-chain CRF parameterisation.

use crate::error::{SeqError, SeqResult};

/// Linear-chain CRF with linear emissions over real-valued features and a full
/// dense transition matrix.
///
/// * `emissions[label*n_features + k]` — emission weight w_k for label
/// * `transitions[prev*n_labels + cur]` — transition weight from `prev` to `cur`
#[derive(Debug, Clone)]
pub struct LinearChainCrf {
    pub n_labels: usize,
    pub n_features: usize,
    pub emissions: Vec<f64>,
    pub transitions: Vec<f64>,
}

impl LinearChainCrf {
    /// Construct a zero-initialised CRF.
    pub fn zeros(n_labels: usize, n_features: usize) -> SeqResult<Self> {
        if n_labels == 0 || n_features == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_labels and n_features must be > 0".to_string(),
            ));
        }
        Ok(Self {
            n_labels,
            n_features,
            emissions: vec![0.0; n_labels * n_features],
            transitions: vec![0.0; n_labels * n_labels],
        })
    }

    /// Number of free parameters (emissions + transitions).
    pub fn param_count(&self) -> usize {
        self.n_labels * self.n_features + self.n_labels * self.n_labels
    }

    /// Pack `(emissions, transitions)` into a flat parameter vector.
    pub fn to_params(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.param_count());
        v.extend_from_slice(&self.emissions);
        v.extend_from_slice(&self.transitions);
        v
    }

    /// Unpack a flat parameter vector into `(emissions, transitions)`.
    pub fn from_params(&mut self, p: &[f64]) -> SeqResult<()> {
        let expected = self.param_count();
        if p.len() != expected {
            return Err(SeqError::ShapeMismatch {
                expected,
                got: p.len(),
            });
        }
        let cut = self.n_labels * self.n_features;
        self.emissions.copy_from_slice(&p[..cut]);
        self.transitions.copy_from_slice(&p[cut..]);
        Ok(())
    }

    /// Compute the emission score for a single (label, feature_vec) pair.
    pub fn emit_score(&self, label: usize, x: &[f64]) -> SeqResult<f64> {
        if label >= self.n_labels {
            return Err(SeqError::IndexOutOfBounds {
                index: label,
                len: self.n_labels,
            });
        }
        if x.len() != self.n_features {
            return Err(SeqError::ShapeMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let mut s = 0.0;
        for (k, &xv) in x.iter().enumerate() {
            s += self.emissions[label * self.n_features + k] * xv;
        }
        Ok(s)
    }

    /// Score for a single (prev_label, cur_label) transition.
    pub fn trans_score(&self, prev: usize, cur: usize) -> SeqResult<f64> {
        if prev >= self.n_labels || cur >= self.n_labels {
            return Err(SeqError::IndexOutOfBounds {
                index: prev.max(cur),
                len: self.n_labels,
            });
        }
        Ok(self.transitions[prev * self.n_labels + cur])
    }

    /// Score of a full label sequence given a feature matrix (T × n_features).
    pub fn sequence_score(&self, x: &[f64], y: &[usize]) -> SeqResult<f64> {
        if y.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        let t_max = y.len();
        if x.len() != t_max * self.n_features {
            return Err(SeqError::ShapeMismatch {
                expected: t_max * self.n_features,
                got: x.len(),
            });
        }
        let mut s = 0.0;
        for t in 0..t_max {
            s += self.emit_score(y[t], &x[t * self.n_features..(t + 1) * self.n_features])?;
            if t > 0 {
                s += self.trans_score(y[t - 1], y[t])?;
            }
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_crf_has_zero_score() {
        let crf = LinearChainCrf::zeros(3, 4).expect("ok");
        let s = crf.sequence_score(&[1.0; 8], &[0, 1]).expect("ok");
        assert!((s).abs() < 1e-12);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let mut crf = LinearChainCrf::zeros(2, 3).expect("ok");
        for v in crf.emissions.iter_mut() {
            *v = 0.5;
        }
        for v in crf.transitions.iter_mut() {
            *v = 1.5;
        }
        let p = crf.to_params();
        let mut crf2 = LinearChainCrf::zeros(2, 3).expect("ok");
        crf2.from_params(&p).expect("ok");
        assert_eq!(crf.emissions, crf2.emissions);
        assert_eq!(crf.transitions, crf2.transitions);
    }
}
