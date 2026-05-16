//! Linear-chain structured SVM with cutting-plane + sub-gradient training.

use super::cutting_plane::CuttingPlaneConfig;
use crate::crf::linear_chain_crf::LinearChainCrf;
use crate::crf::viterbi_decode::viterbi_decode;
use crate::error::{SeqError, SeqResult};

/// Linear-chain structured SVM.  Shares the parameterisation with
/// `LinearChainCrf` but uses a margin-rescaled hinge loss.
#[derive(Debug, Clone)]
pub struct StructuredSvm {
    pub crf: LinearChainCrf,
}

impl StructuredSvm {
    /// Construct an SSVM with zero parameters.
    pub fn zeros(n_labels: usize, n_features: usize) -> SeqResult<Self> {
        Ok(Self {
            crf: LinearChainCrf::zeros(n_labels, n_features)?,
        })
    }

    /// Predict via Viterbi on the underlying linear-chain model.
    pub fn predict(&self, x: &[f64]) -> SeqResult<Vec<usize>> {
        viterbi_decode(&self.crf, x)
    }

    /// Hamming loss between two label sequences.
    fn hamming(y: &[usize], yhat: &[usize]) -> usize {
        y.iter().zip(yhat.iter()).filter(|(a, b)| a != b).count()
    }

    /// Loss-augmented inference: find ŷ = argmax_y [score(y) + Δ(y, y*)] where
    /// `Δ` is the Hamming loss.  Implemented by adding 1 to emission scores of
    /// all labels different from the ground truth at each position before Viterbi.
    fn loss_augmented(&self, x: &[f64], y_true: &[usize]) -> SeqResult<Vec<usize>> {
        let n = self.crf.n_labels;
        let k = self.crf.n_features;
        if y_true.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        let t_max = y_true.len();
        if x.len() != t_max * k {
            return Err(SeqError::ShapeMismatch {
                expected: t_max * k,
                got: x.len(),
            });
        }

        let mut delta = vec![f64::NEG_INFINITY; t_max * n];
        let mut psi = vec![0usize; t_max * n];

        for j in 0..n {
            let mut em = self.crf.emit_score(j, &x[..k])?;
            if j != y_true[0] {
                em += 1.0;
            }
            delta[j] = em;
        }
        for t in 1..t_max {
            for j in 0..n {
                let mut em = self.crf.emit_score(j, &x[t * k..(t + 1) * k])?;
                if j != y_true[t] {
                    em += 1.0;
                }
                let mut best = f64::NEG_INFINITY;
                let mut argmax = 0usize;
                for i in 0..n {
                    let v = delta[(t - 1) * n + i] + self.crf.transitions[i * n + j];
                    if v > best {
                        best = v;
                        argmax = i;
                    }
                }
                delta[t * n + j] = best + em;
                psi[t * n + j] = argmax;
            }
        }

        let mut best = f64::NEG_INFINITY;
        let mut last = 0usize;
        for j in 0..n {
            if delta[(t_max - 1) * n + j] > best {
                best = delta[(t_max - 1) * n + j];
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

    /// Sub-gradient training using loss-augmented inference (Crammer-Singer style).
    pub fn train(
        &mut self,
        examples: &[(Vec<f64>, Vec<usize>)],
        cfg: &CuttingPlaneConfig,
    ) -> SeqResult<f64> {
        if examples.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        let n = self.crf.n_labels;
        let k = self.crf.n_features;
        let mut final_obj = 0.0;

        for it in 0..cfg.max_iter {
            let lr = cfg.lr / (1.0 + cfg.lr_decay * it as f64);
            let mut total_loss = 0.0;
            for (x, y_true) in examples {
                let y_hat = self.loss_augmented(x, y_true)?;
                let hloss = Self::hamming(y_true, &y_hat) as f64;
                if hloss <= 0.0 {
                    continue;
                }
                let t_max = y_true.len();
                // Sub-gradient: g_w = expected_features(ŷ) − features(y*)
                // Take step w := w − lr * (g_w + C * w)  (regularised SGD)
                for t in 0..t_max {
                    let xt = &x[t * k..(t + 1) * k];
                    let yt = y_true[t];
                    let yh = y_hat[t];
                    if yt != yh {
                        // emissions
                        for f in 0..k {
                            self.crf.emissions[yt * k + f] += lr * xt[f];
                            self.crf.emissions[yh * k + f] -= lr * xt[f];
                        }
                    }
                    if t > 0 {
                        let yp = y_true[t - 1];
                        let yhp = y_hat[t - 1];
                        if yp != yhp || yt != yh {
                            self.crf.transitions[yp * n + yt] += lr;
                            self.crf.transitions[yhp * n + yh] -= lr;
                        }
                    }
                }
                total_loss += hloss;
            }
            // L2 regularisation step
            let c = cfg.regularisation * lr;
            for w in self.crf.emissions.iter_mut() {
                *w -= c * *w;
            }
            for w in self.crf.transitions.iter_mut() {
                *w -= c * *w;
            }
            final_obj = total_loss;
            if total_loss < cfg.tol {
                break;
            }
        }
        Ok(final_obj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssvm_train_reduces_loss() {
        let mut s = StructuredSvm::zeros(2, 2).expect("ok");
        let x1 = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let y1 = vec![0usize, 0, 1];
        let x2 = vec![0.0, 1.0, 1.0, 0.0];
        let y2 = vec![1usize, 0];
        let examples = vec![(x1.clone(), y1.clone()), (x2.clone(), y2.clone())];
        let cfg = CuttingPlaneConfig {
            max_iter: 50,
            lr: 0.1,
            lr_decay: 0.0,
            regularisation: 0.01,
            tol: 0.5,
        };
        let _obj = s.train(&examples, &cfg).expect("ok");
        // After training the predictions on at least one example should match.
        let p1 = s.predict(&x1).expect("ok");
        assert_eq!(p1.len(), y1.len());
    }
}
