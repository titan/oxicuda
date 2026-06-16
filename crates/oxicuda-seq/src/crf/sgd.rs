//! CRF training via SGD / AdaGrad.
//!
//! Linear-chain CRF with parameter vector θ = [emissions | transitions]:
//!   - emissions: `[n_tags × n_features]` (row-major: tag is the outer index)
//!   - transitions: `[n_tags × n_tags]` (row-major: `tr[i,j]` = prev-tag i → curr-tag j)
//!
//! Score: Σ_t (`w_emit[y_t]` · x_t) + Σ_{t≥1} `w_tr[y_{t-1}, y_t]`
//!
//! Gradient is computed via log-space forward-backward.
//! AdaGrad: G_i += g_i²;  θ_i -= lr / sqrt(G_i + ε) * g_i  (minimising NLL).

use crate::error::{SeqError, SeqResult};
use crate::handle::LcgRng;

// ─── logsumexp helper ────────────────────────────────────────────────────────

#[inline]
fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let s: f64 = xs.iter().map(|&x| (x - m).exp()).sum();
    m + s.ln()
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for SGD/AdaGrad CRF training.
#[derive(Debug, Clone)]
pub struct CrfSgdConfig {
    /// Number of label tags.
    pub n_tags: usize,
    /// Number of emission features per position.
    pub n_features: usize,
    /// Number of training epochs.
    pub n_epochs: usize,
    /// Base learning rate.
    pub lr: f64,
    /// L2 regularisation strength.
    pub l2_reg: f64,
    /// Whether to use AdaGrad adaptive learning rates.
    pub adagrad: bool,
}

// ─── CrfSgd ──────────────────────────────────────────────────────────────────

/// Linear-chain CRF trained with SGD or AdaGrad.
///
/// Parameter layout:
///   `weights[tag * n_features + feat]`  → emission weight for (tag, feat)
///   `weights[n_tags * n_features + prev * n_tags + curr]`  → transition weight
#[derive(Debug, Clone)]
pub struct CrfSgd {
    /// Full parameter vector `[n_tags × n_features + n_tags × n_tags]`.
    pub weights: Vec<f64>,
    /// Hyper-parameters.
    config: CrfSgdConfig,
    /// Accumulated squared gradients for AdaGrad.
    adagrad_acc: Vec<f64>,
}

impl CrfSgd {
    const ADAGRAD_EPS: f64 = 1e-8;

    // ── Construction ─────────────────────────────────────────────────────────

    /// Create and initialise a new CRF (weights ~ N(0, 0.1)).
    pub fn new(config: CrfSgdConfig, rng: &mut LcgRng) -> SeqResult<Self> {
        if config.n_tags == 0 {
            return Err(SeqError::InvalidConfiguration("n_tags must be > 0".into()));
        }
        if config.n_features == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_features must be > 0".into(),
            ));
        }
        let n_emit = config.n_tags * config.n_features;
        let n_tr = config.n_tags * config.n_tags;
        let n_params = n_emit + n_tr;
        let weights: Vec<f64> = (0..n_params).map(|_| rng.next_normal() * 0.1).collect();
        let adagrad_acc = vec![0.0f64; n_params];
        Ok(Self {
            weights,
            config,
            adagrad_acc,
        })
    }

    // ── Index helpers ─────────────────────────────────────────────────────────

    /// Index into the emission block.
    #[inline]
    fn emit_idx(&self, tag: usize, feat: usize) -> usize {
        tag * self.config.n_features + feat
    }

    /// Index into the transition block.
    #[inline]
    fn tr_idx(&self, prev_tag: usize, curr_tag: usize) -> usize {
        self.config.n_tags * self.config.n_features + prev_tag * self.config.n_tags + curr_tag
    }

    // ── Public accessors ──────────────────────────────────────────────────────

    /// Emission weight for `(tag, feat)`.
    pub fn emission_weight(&self, tag: usize, feat: usize) -> f64 {
        self.weights[self.emit_idx(tag, feat)]
    }

    /// Transition weight for the `prev_tag → curr_tag` edge.
    pub fn transition_weight(&self, prev_tag: usize, curr_tag: usize) -> f64 {
        self.weights[self.tr_idx(prev_tag, curr_tag)]
    }

    // ── Score helpers ─────────────────────────────────────────────────────────

    /// Emission score at position `t` for tag `j`.
    #[inline]
    fn emit_score(&self, j: usize, feat: &[f64]) -> f64 {
        let base = j * self.config.n_features;
        let mut s = 0.0;
        for f in 0..self.config.n_features {
            s += self.weights[base + f] * feat[f];
        }
        s
    }

    // ── Forward algorithm (log-partition) ─────────────────────────────────────

    /// Compute the log-partition function Z for `features[seq_len][n_features]`.
    pub fn log_partition(&self, features: &[Vec<f64>], seq_len: usize) -> SeqResult<f64> {
        if seq_len == 0 {
            return Err(SeqError::EmptyInput);
        }
        if features.len() < seq_len {
            return Err(SeqError::ShapeMismatch {
                expected: seq_len,
                got: features.len(),
            });
        }
        let n = self.config.n_tags;
        let mut alpha = vec![f64::NEG_INFINITY; n];
        // Initialise: α_0(j) = emit(j, x_0)
        for j in 0..n {
            alpha[j] = self.emit_score(j, &features[0]);
        }
        let mut tmp = vec![0.0f64; n];
        for t in 1..seq_len {
            let mut alpha_new = vec![f64::NEG_INFINITY; n];
            for j in 0..n {
                for i in 0..n {
                    tmp[i] = alpha[i] + self.transition_weight(i, j);
                }
                alpha_new[j] = logsumexp(&tmp) + self.emit_score(j, &features[t]);
            }
            alpha = alpha_new;
        }
        Ok(logsumexp(&alpha))
    }

    // ── Forward-backward (all α, β tables) ───────────────────────────────────

    fn forward_table(&self, features: &[Vec<f64>], seq_len: usize) -> Vec<Vec<f64>> {
        let n = self.config.n_tags;
        let mut table = vec![vec![f64::NEG_INFINITY; n]; seq_len];
        for j in 0..n {
            table[0][j] = self.emit_score(j, &features[0]);
        }
        let mut tmp = vec![0.0f64; n];
        for t in 1..seq_len {
            for j in 0..n {
                for i in 0..n {
                    tmp[i] = table[t - 1][i] + self.transition_weight(i, j);
                }
                table[t][j] = logsumexp(&tmp) + self.emit_score(j, &features[t]);
            }
        }
        table
    }

    fn backward_table(&self, features: &[Vec<f64>], seq_len: usize) -> Vec<Vec<f64>> {
        let n = self.config.n_tags;
        let mut table = vec![vec![0.0f64; n]; seq_len]; // β_{T-1}(i) = 0 in log-space
        let mut tmp = vec![0.0f64; n];
        for t in (0..seq_len - 1).rev() {
            for i in 0..n {
                for j in 0..n {
                    tmp[j] = self.transition_weight(i, j)
                        + self.emit_score(j, &features[t + 1])
                        + table[t + 1][j];
                }
                table[t][i] = logsumexp(&tmp);
            }
        }
        table
    }

    // ── Gradient computation ──────────────────────────────────────────────────

    /// Compute gradient (NLL direction, i.e. ascent on NLL = descent on LL) and
    /// negative log-likelihood for one sequence.
    ///
    /// Returns `(nll, gradient)`.
    fn gradient_one(&self, features: &[Vec<f64>], labels: &[usize]) -> SeqResult<(f64, Vec<f64>)> {
        let seq_len = labels.len();
        if seq_len == 0 {
            return Err(SeqError::EmptyInput);
        }
        let n = self.config.n_tags;
        let k = self.config.n_features;
        let n_params = self.weights.len();

        // Validate labels.
        for (t, &y) in labels.iter().enumerate() {
            if y >= n {
                return Err(SeqError::IndexOutOfBounds { index: y, len: n });
            }
            if features[t].len() != k {
                return Err(SeqError::ShapeMismatch {
                    expected: k,
                    got: features[t].len(),
                });
            }
        }

        // ── Log-partition (forward only) ──────────────────────────────────────
        let alpha = self.forward_table(features, seq_len);
        let log_z = logsumexp(&alpha[seq_len - 1]);

        // ── Backward ──────────────────────────────────────────────────────────
        let beta = self.backward_table(features, seq_len);

        // ── Score of true path ────────────────────────────────────────────────
        let mut score_true = self.emit_score(labels[0], &features[0]);
        for t in 1..seq_len {
            score_true += self.transition_weight(labels[t - 1], labels[t])
                + self.emit_score(labels[t], &features[t]);
        }
        let nll = log_z - score_true;

        // ── Gradient ──────────────────────────────────────────────────────────
        // grad[θ] = E_model[f(x,y)] - f(x, y_true)
        let mut grad = vec![0.0f64; n_params];

        // Emission expected counts via γ_t(j) = α_t(j) + β_t(j) - log_z
        for t in 0..seq_len {
            let feat = &features[t];
            for j in 0..n {
                let log_gamma = alpha[t][j] + beta[t][j] - log_z;
                let gamma = log_gamma.exp();
                let base = self.emit_idx(j, 0);
                for f in 0..k {
                    grad[base + f] += gamma * feat[f];
                }
            }
        }
        // Subtract true emission counts
        for t in 0..seq_len {
            let feat = &features[t];
            let j = labels[t];
            let base = self.emit_idx(j, 0);
            for f in 0..k {
                grad[base + f] -= feat[f];
            }
        }

        // Transition expected counts via ξ_{t}(i,j):
        // log ξ_t(i,j) = α_t(i) + tr(i,j) + emit_{t+1}(j) + β_{t+1}(j) - log_z
        for t in 0..seq_len - 1 {
            for i in 0..n {
                for j in 0..n {
                    let log_xi = alpha[t][i]
                        + self.transition_weight(i, j)
                        + self.emit_score(j, &features[t + 1])
                        + beta[t + 1][j]
                        - log_z;
                    let xi = log_xi.exp();
                    grad[self.tr_idx(i, j)] += xi;
                }
            }
        }
        // Subtract true transition counts
        for t in 1..seq_len {
            let (i, j) = (labels[t - 1], labels[t]);
            grad[self.tr_idx(i, j)] -= 1.0;
        }

        Ok((nll, grad))
    }

    // ── Parameter update ──────────────────────────────────────────────────────

    /// Apply one SGD / AdaGrad step given a pre-computed gradient vector.
    fn apply_update(&mut self, grad: &[f64]) {
        let lr = self.config.lr;
        let eps = Self::ADAGRAD_EPS;
        let n_params = self.weights.len();

        if self.config.adagrad {
            for i in 0..n_params {
                self.adagrad_acc[i] += grad[i] * grad[i];
                let eff_lr = lr / (self.adagrad_acc[i] + eps).sqrt();
                self.weights[i] -= eff_lr * grad[i];
            }
        } else {
            for i in 0..n_params {
                self.weights[i] -= lr * grad[i];
            }
        }
    }

    // ── Public training API ───────────────────────────────────────────────────

    /// Compute gradient for one sequence and update weights in-place.
    ///
    /// Returns the negative log-likelihood for this sample.
    pub fn update_one(&mut self, features: &[Vec<f64>], labels: &[usize]) -> SeqResult<f64> {
        let (nll, mut grad) = self.gradient_one(features, labels)?;
        // Add L2 regularisation gradient
        let l2 = self.config.l2_reg;
        if l2 > 0.0 {
            for i in 0..self.weights.len() {
                grad[i] += l2 * self.weights[i];
            }
        }
        self.apply_update(&grad);
        Ok(nll)
    }

    /// Train for `n_epochs`, returning average NLL per epoch.
    pub fn fit(
        &mut self,
        all_features: &[Vec<Vec<f64>>],
        all_labels: &[Vec<usize>],
    ) -> SeqResult<Vec<f64>> {
        if all_features.len() != all_labels.len() {
            return Err(SeqError::LengthMismatch {
                a: all_features.len(),
                b: all_labels.len(),
            });
        }
        let n_samples = all_features.len();
        if n_samples == 0 {
            return Err(SeqError::EmptyInput);
        }
        let mut epoch_losses = Vec::with_capacity(self.config.n_epochs);
        for _epoch in 0..self.config.n_epochs {
            let mut total_nll = 0.0;
            for s in 0..n_samples {
                total_nll += self.update_one(&all_features[s], &all_labels[s])?;
            }
            epoch_losses.push(total_nll / n_samples as f64);
        }
        Ok(epoch_losses)
    }

    // ── Decoding ──────────────────────────────────────────────────────────────

    /// Viterbi decode — returns the best tag sequence for `features`.
    pub fn decode(&self, features: &[Vec<f64>], seq_len: usize) -> SeqResult<Vec<usize>> {
        if seq_len == 0 {
            return Err(SeqError::EmptyInput);
        }
        if features.len() < seq_len {
            return Err(SeqError::ShapeMismatch {
                expected: seq_len,
                got: features.len(),
            });
        }
        let n = self.config.n_tags;
        let mut viterbi = vec![f64::NEG_INFINITY; n];
        let mut backptr = vec![vec![0usize; n]; seq_len];

        // Initialise
        for j in 0..n {
            viterbi[j] = self.emit_score(j, &features[0]);
        }

        // Fill DP
        for t in 1..seq_len {
            let mut viterbi_new = vec![f64::NEG_INFINITY; n];
            for j in 0..n {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_prev = 0;
                for i in 0..n {
                    let s = viterbi[i] + self.transition_weight(i, j);
                    if s > best_score {
                        best_score = s;
                        best_prev = i;
                    }
                }
                viterbi_new[j] = best_score + self.emit_score(j, &features[t]);
                backptr[t][j] = best_prev;
            }
            viterbi = viterbi_new;
        }

        // Find best last tag
        let mut best_last = 0;
        let mut best_val = f64::NEG_INFINITY;
        for j in 0..n {
            if viterbi[j] > best_val {
                best_val = viterbi[j];
                best_last = j;
            }
        }

        // Backtrace
        let mut path = vec![0usize; seq_len];
        path[seq_len - 1] = best_last;
        for t in (0..seq_len - 1).rev() {
            path[t] = backptr[t + 1][path[t + 1]];
        }
        Ok(path)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(adagrad: bool) -> CrfSgdConfig {
        CrfSgdConfig {
            n_tags: 3,
            n_features: 4,
            n_epochs: 5,
            lr: 0.05,
            l2_reg: 1e-4,
            adagrad,
        }
    }

    fn make_crf(adagrad: bool) -> CrfSgd {
        let mut rng = LcgRng::new(42);
        CrfSgd::new(make_config(adagrad), &mut rng).expect("construction failed")
    }

    fn simple_data(n_tags: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
        // Sequence of length 3 with deterministic features
        let features = vec![
            vec![1.0, 0.0, 0.5, -0.5],
            vec![0.0, 1.0, -0.5, 0.5],
            vec![0.5, 0.5, 0.0, 1.0],
        ];
        // Keep only as many features as needed
        let features: Vec<Vec<f64>> = features
            .into_iter()
            .map(|f| f.into_iter().take(n_features).collect())
            .collect();
        let labels = vec![0, 1 % n_tags, 2 % n_tags];
        (features, labels)
    }

    #[test]
    fn weights_shape() {
        let crf = make_crf(false);
        assert_eq!(
            crf.weights.len(),
            3 * 4 + 3 * 3,
            "weights.len() should be n_tags*n_features + n_tags*n_tags"
        );
    }

    #[test]
    fn decode_output_len() {
        let crf = make_crf(false);
        let (features, _) = simple_data(3, 4);
        let seq_len = features.len();
        let path = crf.decode(&features, seq_len).expect("decode failed");
        assert_eq!(path.len(), seq_len);
    }

    #[test]
    fn decode_valid_tags() {
        let crf = make_crf(false);
        let (features, _) = simple_data(3, 4);
        let seq_len = features.len();
        let path = crf.decode(&features, seq_len).expect("decode failed");
        for &tag in &path {
            assert!(tag < 3, "decoded tag {tag} >= n_tags=3");
        }
    }

    #[test]
    fn log_partition_finite() {
        let crf = make_crf(false);
        let (features, _) = simple_data(3, 4);
        let lz = crf
            .log_partition(&features, features.len())
            .expect("lz failed");
        assert!(lz.is_finite(), "log_partition should be finite, got {lz}");
    }

    #[test]
    fn update_decreases_loss() {
        let mut rng = LcgRng::new(7);
        let mut config = make_config(true);
        config.n_epochs = 30;
        config.lr = 0.1;
        config.n_features = 4;
        config.n_tags = 3;
        let mut crf = CrfSgd::new(config, &mut rng).expect("new failed");

        // Build a simple dataset: 4 samples, each of seq_len=3
        let all_feats: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|seed| {
                let mut r = LcgRng::new(seed as u64 + 1);
                (0..3)
                    .map(|_| (0..4).map(|_| r.next_normal()).collect())
                    .collect()
            })
            .collect();
        let all_labels: Vec<Vec<usize>> =
            vec![vec![0, 1, 2], vec![2, 0, 1], vec![1, 2, 0], vec![0, 0, 1]];
        let losses = crf.fit(&all_feats, &all_labels).expect("fit failed");
        assert!(!losses.is_empty());
        // Average of last 5 epochs < average of first 5
        let first =
            losses[..5.min(losses.len())].iter().sum::<f64>() / 5.0_f64.min(losses.len() as f64);
        let last_start = losses.len().saturating_sub(5);
        let last = losses[last_start..].iter().sum::<f64>() / (losses.len() - last_start) as f64;
        assert!(
            last < first,
            "loss did not decrease: first={first:.4}, last={last:.4}"
        );
    }

    #[test]
    fn adagrad_different_from_sgd() {
        let mut rng_sgd = LcgRng::new(42);
        let mut rng_ada = LcgRng::new(42);
        let mut config_sgd = make_config(false);
        let mut config_ada = make_config(true);
        config_sgd.n_epochs = 5;
        config_ada.n_epochs = 5;
        let mut crf_sgd = CrfSgd::new(config_sgd, &mut rng_sgd).expect("new failed");
        let mut crf_ada = CrfSgd::new(config_ada, &mut rng_ada).expect("new failed");

        let (features, labels) = simple_data(3, 4);
        let all_feats = vec![features.clone()];
        let all_labels = vec![labels.clone()];
        crf_sgd.fit(&all_feats, &all_labels).expect("fit sgd");
        crf_ada.fit(&all_feats, &all_labels).expect("fit ada");
        let diff: f64 = crf_sgd
            .weights
            .iter()
            .zip(&crf_ada.weights)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-12, "adagrad and sgd produced identical weights");
    }

    #[test]
    fn viterbi_agrees_with_exhaustive() {
        // For seq_len=2, n_tags=2: exhaustively check all 4 paths.
        let mut rng = LcgRng::new(99);
        let config = CrfSgdConfig {
            n_tags: 2,
            n_features: 3,
            n_epochs: 1,
            lr: 0.01,
            l2_reg: 0.0,
            adagrad: false,
        };
        let crf = CrfSgd::new(config, &mut rng).expect("new");
        let features = vec![vec![1.0, -1.0, 0.5], vec![-0.5, 0.5, 1.0]];
        let path = crf.decode(&features, 2).expect("decode");
        // Enumerate all 4 paths
        let score_path = |y0: usize, y1: usize| -> f64 {
            crf.emit_score(y0, &features[0])
                + crf.transition_weight(y0, y1)
                + crf.emit_score(y1, &features[1])
        };
        let mut best_score = f64::NEG_INFINITY;
        let mut best_path = (0, 0);
        for y0 in 0..2 {
            for y1 in 0..2 {
                let s = score_path(y0, y1);
                if s > best_score {
                    best_score = s;
                    best_path = (y0, y1);
                }
            }
        }
        assert_eq!(path[0], best_path.0, "Viterbi y0 mismatch");
        assert_eq!(path[1], best_path.1, "Viterbi y1 mismatch");
    }

    #[test]
    fn emission_weight_correct() {
        let crf = make_crf(false);
        for tag in 0..3 {
            for feat in 0..4 {
                let expected = crf.weights[tag * 4 + feat];
                assert_eq!(
                    crf.emission_weight(tag, feat),
                    expected,
                    "emission_weight({tag},{feat}) mismatch"
                );
            }
        }
    }

    #[test]
    fn n_tags_zero_error() {
        let mut rng = LcgRng::new(1);
        let config = CrfSgdConfig {
            n_tags: 0,
            n_features: 4,
            n_epochs: 1,
            lr: 0.01,
            l2_reg: 0.0,
            adagrad: false,
        };
        assert!(
            CrfSgd::new(config, &mut rng).is_err(),
            "n_tags=0 should fail"
        );
    }

    #[test]
    fn empty_sequence_error() {
        let crf = make_crf(false);
        let result = crf.decode(&[], 0);
        assert!(result.is_err(), "decode on empty should fail");
    }
}
