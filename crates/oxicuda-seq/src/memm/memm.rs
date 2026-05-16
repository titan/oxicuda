//! Maximum-Entropy Markov Model implementation.

use crate::error::{SeqError, SeqResult};

/// Maximum-entropy Markov model: per-current-label softmax conditioned on the
/// previous label and the current feature vector.
///
/// * `weights[prev * n_labels * n_features + cur * n_features + f]` — weight of
///   feature `f` for the transition `prev -> cur`.
#[derive(Debug, Clone)]
pub struct Memm {
    pub n_labels: usize,
    pub n_features: usize,
    pub weights: Vec<f64>,
    pub start_label: usize,
}

impl Memm {
    /// Zero-initialised MEMM.
    pub fn zeros(n_labels: usize, n_features: usize) -> SeqResult<Self> {
        if n_labels == 0 || n_features == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_labels and n_features must be > 0".to_string(),
            ));
        }
        Ok(Self {
            n_labels,
            n_features,
            weights: vec![0.0; n_labels * n_labels * n_features],
            start_label: 0,
        })
    }

    /// Compute `p(cur | prev, x)` for all `cur` given `prev` and `x`.
    pub fn class_probs(&self, prev: usize, x: &[f64]) -> SeqResult<Vec<f64>> {
        if prev >= self.n_labels {
            return Err(SeqError::IndexOutOfBounds {
                index: prev,
                len: self.n_labels,
            });
        }
        if x.len() != self.n_features {
            return Err(SeqError::ShapeMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let mut logits = vec![0.0; self.n_labels];
        let base = prev * self.n_labels * self.n_features;
        for cur in 0..self.n_labels {
            let row =
                &self.weights[base + cur * self.n_features..base + (cur + 1) * self.n_features];
            let s: f64 = row.iter().zip(x.iter()).map(|(w, v)| w * v).sum();
            logits[cur] = s;
        }
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
        let z: f64 = exps.iter().sum();
        Ok(exps.iter().map(|e| e / z).collect())
    }

    /// Greedy decode: pick argmax at each step.
    pub fn decode_greedy(&self, x: &[f64]) -> SeqResult<Vec<usize>> {
        if x.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if x.len() % self.n_features != 0 {
            return Err(SeqError::DimensionMismatch {
                a: x.len(),
                b: self.n_features,
            });
        }
        let t_max = x.len() / self.n_features;
        let mut path = Vec::with_capacity(t_max);
        let mut prev = self.start_label;
        for t in 0..t_max {
            let probs =
                self.class_probs(prev, &x[t * self.n_features..(t + 1) * self.n_features])?;
            let (best, _) =
                probs
                    .iter()
                    .enumerate()
                    .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv { (i, v) } else { (bi, bv) }
                    });
            path.push(best);
            prev = best;
        }
        Ok(path)
    }

    /// Beam decode of width `beam`.
    pub fn decode_beam(&self, x: &[f64], beam: usize) -> SeqResult<Vec<usize>> {
        if beam == 0 {
            return Err(SeqError::InvalidConfiguration(
                "beam width must be > 0".to_string(),
            ));
        }
        if x.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if x.len() % self.n_features != 0 {
            return Err(SeqError::DimensionMismatch {
                a: x.len(),
                b: self.n_features,
            });
        }
        let t_max = x.len() / self.n_features;
        // Each beam item: (log_prob, path)
        let mut beam_items: Vec<(f64, Vec<usize>)> = vec![(0.0, Vec::new())];
        for t in 0..t_max {
            let mut new_items: Vec<(f64, Vec<usize>)> =
                Vec::with_capacity(beam_items.len() * self.n_labels);
            for (lp, path) in &beam_items {
                let prev = path.last().copied().unwrap_or(self.start_label);
                let probs =
                    self.class_probs(prev, &x[t * self.n_features..(t + 1) * self.n_features])?;
                for (cur, &p) in probs.iter().enumerate() {
                    let logp = if p > 0.0 { p.ln() } else { f64::NEG_INFINITY };
                    let mut new_path = path.clone();
                    new_path.push(cur);
                    new_items.push((lp + logp, new_path));
                }
            }
            new_items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            new_items.truncate(beam);
            beam_items = new_items;
        }
        let best = beam_items
            .into_iter()
            .next()
            .ok_or_else(|| SeqError::NumericalInstability("empty beam".to_string()))?;
        Ok(best.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_probs_sum_to_one() {
        let m = Memm::zeros(3, 2).expect("ok");
        let p = m.class_probs(0, &[1.0, 1.0]).expect("ok");
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
        for &v in &p {
            assert!((v - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn greedy_zero_weights() {
        let m = Memm::zeros(2, 2).expect("ok");
        let path = m.decode_greedy(&[1.0, 0.0, 0.0, 1.0]).expect("ok");
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn beam_matches_greedy_zero_weights() {
        let m = Memm::zeros(2, 2).expect("ok");
        let g = m.decode_greedy(&[1.0, 0.0, 0.0, 1.0]).expect("ok");
        let b = m.decode_beam(&[1.0, 0.0, 0.0, 1.0], 4).expect("ok");
        assert_eq!(g.len(), b.len());
    }
}
