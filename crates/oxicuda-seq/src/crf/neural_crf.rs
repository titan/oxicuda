//! Neural linear-chain Conditional Random Field.
//!
//! Reference: Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K.
//! & Kuksa, P. (2011). *Natural Language Processing (Almost) from Scratch*.
//! JMLR 12, 2493–2537 — the "sentence-level log-likelihood" (SLL) network, which
//! couples a neural feature extractor with a CRF-style transition matrix and
//! trains end-to-end with the forward (partition-function) algorithm.
//!
//! # Model
//!
//! A linear-chain CRF whose **emission / unary scores come from a multilayer
//! perceptron (MLP)** over per-position input features rather than from a linear
//! score over sparse features:
//!
//! ```text
//! h_t   = tanh(W1 · x_t + b1)              (hidden activations, per position t)
//! e_t   = W2 · h_t + b2                    (emission scores, one per tag k)
//! ```
//!
//! together with a learned `K × K` transition matrix `A` (`A[i][j]` = score of a
//! transition from tag `i` to tag `j`). The score of a full tag sequence is
//!
//! ```text
//! s(y, x) = Σ_t e_t[y_t]  +  Σ_{t>0} A[y_{t-1}][y_t]
//! ```
//!
//! and the conditional probability is `p(y|x) = exp(s(y, x)) / Z(x)` where the
//! log-partition `log Z(x)` is computed by the **forward algorithm in log-space**
//! (log-sum-exp). Decoding uses **Viterbi**. Training minimises the negative
//! log-likelihood (NLL); the emission gradient
//! `∂NLL/∂e_t[k] = p(y_t = k | x) − 1[gold_t = k]` (from forward–backward
//! marginals) is back-propagated through the MLP, and
//! `∂NLL/∂A[i][j] = Σ_t p(y_{t-1}=i, y_t=j | x) − count_gold(i → j)`.
//!
//! The CRF inference layer (log-space forward, backward, Viterbi) is implemented
//! here directly on the dense emission tensor `e[t][k]`, mirroring the score-space
//! forward–backward used by [`crate::crf::crf_train`] but operating on neural
//! emissions instead of linear feature scores.
//!
//! Production code never panics: every fallible path validates its inputs and
//! returns [`SeqError`].

use crate::error::{SeqError, SeqResult};
use crate::handle::LcgRng;
use crate::hmm::forward_backward::logsumexp;

/// A neural linear-chain CRF: an MLP emission scorer plus a `K × K` transition
/// matrix, trained end-to-end against the CRF negative log-likelihood.
///
/// Parameter layout (all row-major, `f64`):
///
/// * `w1[h * input_dim + d]` — input→hidden weight (`hidden_dim × input_dim`)
/// * `b1[h]` — hidden bias (`hidden_dim`)
/// * `w2[k * hidden_dim + h]` — hidden→tag weight (`n_tags × hidden_dim`)
/// * `b2[k]` — tag bias (`n_tags`)
/// * `transitions[i * n_tags + j]` — transition score `i → j` (`n_tags × n_tags`)
#[derive(Debug, Clone)]
pub struct NeuralCrf {
    /// Number of output tags (`K`).
    pub n_tags: usize,
    /// Dimensionality of the per-position input feature vector.
    pub input_dim: usize,
    /// Hidden-layer width of the emission MLP.
    pub hidden_dim: usize,
    /// Input→hidden weight matrix (`hidden_dim × input_dim`).
    pub w1: Vec<f64>,
    /// Hidden bias (`hidden_dim`).
    pub b1: Vec<f64>,
    /// Hidden→tag weight matrix (`n_tags × hidden_dim`).
    pub w2: Vec<f64>,
    /// Tag bias (`n_tags`).
    pub b2: Vec<f64>,
    /// Transition score matrix (`n_tags × n_tags`), row = previous tag.
    pub transitions: Vec<f64>,
}

/// Gradients of the NLL with respect to every parameter of a [`NeuralCrf`].
///
/// Field shapes mirror the corresponding [`NeuralCrf`] parameter arrays exactly.
#[derive(Debug, Clone)]
pub struct NeuralCrfGrad {
    /// Gradient w.r.t. `w1`.
    pub w1: Vec<f64>,
    /// Gradient w.r.t. `b1`.
    pub b1: Vec<f64>,
    /// Gradient w.r.t. `w2`.
    pub w2: Vec<f64>,
    /// Gradient w.r.t. `b2`.
    pub b2: Vec<f64>,
    /// Gradient w.r.t. `transitions`.
    pub transitions: Vec<f64>,
}

/// Cached intermediates from a forward pass, reused by the backward pass.
///
/// Holds the per-position hidden activations and emission scores so the backward
/// pass can back-propagate the emission gradient through the MLP without a second
/// forward evaluation.
#[derive(Debug, Clone)]
pub struct NeuralCrfForward {
    /// Number of positions `T`.
    pub t_max: usize,
    /// Hidden activations after `tanh`, flattened `T × hidden_dim`.
    pub hidden: Vec<f64>,
    /// Emission scores `e[t][k]`, flattened `T × n_tags`.
    pub emit: Vec<f64>,
}

impl NeuralCrf {
    /// Construct a zero-initialised neural CRF.
    ///
    /// All dimensions must be positive; otherwise [`SeqError::InvalidConfiguration`].
    pub fn zeros(n_tags: usize, input_dim: usize, hidden_dim: usize) -> SeqResult<Self> {
        if n_tags == 0 || input_dim == 0 || hidden_dim == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_tags, input_dim and hidden_dim must all be > 0".to_string(),
            ));
        }
        Ok(Self {
            n_tags,
            input_dim,
            hidden_dim,
            w1: vec![0.0; hidden_dim * input_dim],
            b1: vec![0.0; hidden_dim],
            w2: vec![0.0; n_tags * hidden_dim],
            b2: vec![0.0; n_tags],
            transitions: vec![0.0; n_tags * n_tags],
        })
    }

    /// Construct a neural CRF with small random weights drawn from a seeded LCG.
    ///
    /// Weights are sampled `~ U(-scale, scale)`; biases and the transition matrix
    /// start at zero. `scale` must be finite and positive.
    pub fn new(
        n_tags: usize,
        input_dim: usize,
        hidden_dim: usize,
        scale: f64,
        rng: &mut LcgRng,
    ) -> SeqResult<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(SeqError::InvalidParameter {
                name: "scale".to_string(),
                value: scale,
            });
        }
        let mut net = Self::zeros(n_tags, input_dim, hidden_dim)?;
        for v in net.w1.iter_mut() {
            *v = rng.next_range(-scale, scale);
        }
        for v in net.w2.iter_mut() {
            *v = rng.next_range(-scale, scale);
        }
        Ok(net)
    }

    /// Total number of free parameters.
    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len() + self.transitions.len()
    }

    /// Validate that an input feature buffer has the expected length for `t_max`
    /// positions and return `t_max`.
    fn check_input(&self, x: &[f64]) -> SeqResult<usize> {
        if x.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if x.len() % self.input_dim != 0 {
            return Err(SeqError::DimensionMismatch {
                a: x.len(),
                b: self.input_dim,
            });
        }
        Ok(x.len() / self.input_dim)
    }

    /// Run the MLP emission scorer over an input feature matrix `x`
    /// (`T × input_dim`, row-major), returning cached hidden activations and the
    /// emission tensor `e[t][k]` (`T × n_tags`).
    pub fn forward(&self, x: &[f64]) -> SeqResult<NeuralCrfForward> {
        let t_max = self.check_input(x)?;
        let d = self.input_dim;
        let hh = self.hidden_dim;
        let k = self.n_tags;
        let mut hidden = vec![0.0; t_max * hh];
        let mut emit = vec![0.0; t_max * k];
        for t in 0..t_max {
            let xt = &x[t * d..(t + 1) * d];
            // Hidden layer: h = tanh(W1 x + b1)
            for h in 0..hh {
                let mut acc = self.b1[h];
                let row = h * d;
                for (dd, &xv) in xt.iter().enumerate() {
                    acc += self.w1[row + dd] * xv;
                }
                hidden[t * hh + h] = acc.tanh();
            }
            // Output layer: e = W2 h + b2
            for tag in 0..k {
                let mut acc = self.b2[tag];
                let row = tag * hh;
                for h in 0..hh {
                    acc += self.w2[row + h] * hidden[t * hh + h];
                }
                emit[t * k + tag] = acc;
            }
        }
        Ok(NeuralCrfForward {
            t_max,
            hidden,
            emit,
        })
    }

    /// Score of a full tag sequence given emission scores: Σ emissions + Σ transitions.
    fn sequence_score(&self, emit: &[f64], y: &[usize]) -> SeqResult<f64> {
        let k = self.n_tags;
        let t_max = y.len();
        if t_max == 0 {
            return Err(SeqError::EmptyInput);
        }
        if emit.len() != t_max * k {
            return Err(SeqError::ShapeMismatch {
                expected: t_max * k,
                got: emit.len(),
            });
        }
        let mut s = 0.0;
        for t in 0..t_max {
            let yt = y[t];
            if yt >= k {
                return Err(SeqError::IndexOutOfBounds { index: yt, len: k });
            }
            s += emit[t * k + yt];
            if t > 0 {
                s += self.transitions[y[t - 1] * k + yt];
            }
        }
        Ok(s)
    }

    /// Log-partition `log Z(x)` via the forward algorithm in log-space.
    ///
    /// `alpha_t(j) = logsumexp_i(alpha_{t-1}(i) + A[i][j]) + e_t[j]`,
    /// `log Z = logsumexp_j alpha_{T-1}(j)`.
    pub fn log_partition(&self, emit: &[f64]) -> SeqResult<f64> {
        let alpha = self.forward_scores(emit)?;
        let k = self.n_tags;
        let t_max = emit.len() / k;
        Ok(logsumexp(&alpha[(t_max - 1) * k..]))
    }

    /// Forward log-scores `alpha[t][j]` over the dense emission tensor.
    fn forward_scores(&self, emit: &[f64]) -> SeqResult<Vec<f64>> {
        let k = self.n_tags;
        if emit.is_empty() || emit.len() % k != 0 {
            return Err(SeqError::DimensionMismatch {
                a: emit.len(),
                b: k,
            });
        }
        let t_max = emit.len() / k;
        let mut alpha = vec![f64::NEG_INFINITY; t_max * k];
        alpha[..k].copy_from_slice(&emit[..k]);
        let mut tmp = vec![0.0; k];
        for t in 1..t_max {
            for j in 0..k {
                for i in 0..k {
                    tmp[i] = alpha[(t - 1) * k + i] + self.transitions[i * k + j];
                }
                alpha[t * k + j] = logsumexp(&tmp) + emit[t * k + j];
            }
        }
        Ok(alpha)
    }

    /// Backward log-scores `beta[t][i]` over the dense emission tensor.
    fn backward_scores(&self, emit: &[f64]) -> Vec<f64> {
        let k = self.n_tags;
        let t_max = emit.len() / k;
        let mut beta = vec![0.0; t_max * k];
        let mut tmp = vec![0.0; k];
        for t in (0..t_max.saturating_sub(1)).rev() {
            for i in 0..k {
                for j in 0..k {
                    tmp[j] =
                        self.transitions[i * k + j] + emit[(t + 1) * k + j] + beta[(t + 1) * k + j];
                }
                beta[t * k + i] = logsumexp(&tmp);
            }
        }
        beta
    }

    /// Negative log-likelihood `−log p(y | x) = log Z(x) − s(y, x)` from a cached
    /// forward pass.
    pub fn nll_from_forward(&self, fwd: &NeuralCrfForward, y: &[usize]) -> SeqResult<f64> {
        if y.len() != fwd.t_max {
            return Err(SeqError::LengthMismatch {
                a: y.len(),
                b: fwd.t_max,
            });
        }
        let score = self.sequence_score(&fwd.emit, y)?;
        let log_z = self.log_partition(&fwd.emit)?;
        Ok(log_z - score)
    }

    /// Negative log-likelihood of a gold tag sequence given input features `x`.
    pub fn nll(&self, x: &[f64], y: &[usize]) -> SeqResult<f64> {
        let fwd = self.forward(x)?;
        self.nll_from_forward(&fwd, y)
    }

    /// Decode the highest-scoring tag path with Viterbi over the emission tensor.
    pub fn decode(&self, x: &[f64]) -> SeqResult<Vec<usize>> {
        let fwd = self.forward(x)?;
        self.viterbi(&fwd.emit)
    }

    /// Viterbi decoding directly on a dense emission tensor `e[t][k]`.
    fn viterbi(&self, emit: &[f64]) -> SeqResult<Vec<usize>> {
        let k = self.n_tags;
        if emit.is_empty() || emit.len() % k != 0 {
            return Err(SeqError::DimensionMismatch {
                a: emit.len(),
                b: k,
            });
        }
        let t_max = emit.len() / k;
        let mut delta = vec![f64::NEG_INFINITY; t_max * k];
        let mut psi = vec![0usize; t_max * k];
        delta[..k].copy_from_slice(&emit[..k]);
        for t in 1..t_max {
            for j in 0..k {
                let mut best = f64::NEG_INFINITY;
                let mut argmax = 0usize;
                for i in 0..k {
                    let v = delta[(t - 1) * k + i] + self.transitions[i * k + j];
                    if v > best {
                        best = v;
                        argmax = i;
                    }
                }
                delta[t * k + j] = best + emit[t * k + j];
                psi[t * k + j] = argmax;
            }
        }
        let mut best = f64::NEG_INFINITY;
        let mut last = 0usize;
        for j in 0..k {
            let v = delta[(t_max - 1) * k + j];
            if v > best {
                best = v;
                last = j;
            }
        }
        let mut path = vec![0usize; t_max];
        path[t_max - 1] = last;
        for t in (1..t_max).rev() {
            path[t - 1] = psi[t * k + path[t]];
        }
        Ok(path)
    }

    /// Node and edge posterior marginals from forward–backward.
    ///
    /// Returns `(p_node, p_edge)` where `p_node[t][j] = p(y_t = j | x)`
    /// (`T × n_tags`) and `p_edge[t][i][j] = p(y_t = i, y_{t+1} = j | x)`
    /// (`(T−1) × n_tags × n_tags`).
    fn marginals(&self, emit: &[f64]) -> SeqResult<(Vec<f64>, Vec<f64>)> {
        let k = self.n_tags;
        let alpha = self.forward_scores(emit)?;
        let beta = self.backward_scores(emit);
        let t_max = emit.len() / k;
        let log_z = logsumexp(&alpha[(t_max - 1) * k..]);

        let mut p_node = vec![0.0; t_max * k];
        for t in 0..t_max {
            for j in 0..k {
                p_node[t * k + j] = (alpha[t * k + j] + beta[t * k + j] - log_z).exp();
            }
            let s: f64 = p_node[t * k..t * k + k].iter().sum();
            if s > 0.0 {
                for v in p_node[t * k..t * k + k].iter_mut() {
                    *v /= s;
                }
            }
        }

        let edges = t_max.saturating_sub(1);
        let mut p_edge = vec![0.0; edges * k * k];
        for t in 0..edges {
            let mut s = 0.0;
            for i in 0..k {
                for j in 0..k {
                    let v = (alpha[t * k + i]
                        + self.transitions[i * k + j]
                        + emit[(t + 1) * k + j]
                        + beta[(t + 1) * k + j]
                        - log_z)
                        .exp();
                    p_edge[t * k * k + i * k + j] = v;
                    s += v;
                }
            }
            if s > 0.0 {
                for v in p_edge[t * k * k..(t + 1) * k * k].iter_mut() {
                    *v /= s;
                }
            }
        }
        Ok((p_node, p_edge))
    }

    /// Back-propagate the NLL gradient given a cached forward pass and the gold tags.
    ///
    /// Returns the NLL and a [`NeuralCrfGrad`]. The emission gradient
    /// `g_e[t][k] = p(y_t = k | x) − 1[gold_t = k]` is back-propagated through the
    /// MLP (`tanh` derivative `1 − h²`); the transition gradient is
    /// `Σ_t p(y_{t-1}=i, y_t=j | x) − count_gold(i → j)`.
    pub fn backward(
        &self,
        x: &[f64],
        fwd: &NeuralCrfForward,
        y: &[usize],
    ) -> SeqResult<(f64, NeuralCrfGrad)> {
        let t_max = self.check_input(x)?;
        if t_max != fwd.t_max {
            return Err(SeqError::LengthMismatch {
                a: t_max,
                b: fwd.t_max,
            });
        }
        if y.len() != t_max {
            return Err(SeqError::LengthMismatch {
                a: y.len(),
                b: t_max,
            });
        }
        let k = self.n_tags;
        let hh = self.hidden_dim;
        let d = self.input_dim;
        for &yt in y {
            if yt >= k {
                return Err(SeqError::IndexOutOfBounds { index: yt, len: k });
            }
        }

        let (p_node, p_edge) = self.marginals(&fwd.emit)?;
        let nll = self.nll_from_forward(fwd, y)?;

        // Emission-score gradient: g_e[t][k] = p(y_t = k) − 1[gold_t = k].
        let mut g_emit = p_node.clone();
        for t in 0..t_max {
            g_emit[t * k + y[t]] -= 1.0;
        }

        // Transition gradient: Σ_t p_edge[t][i][j] − count_gold(i → j).
        let mut g_trans = vec![0.0; k * k];
        for t in 0..t_max.saturating_sub(1) {
            for i in 0..k {
                for j in 0..k {
                    g_trans[i * k + j] += p_edge[t * k * k + i * k + j];
                }
            }
            g_trans[y[t] * k + y[t + 1]] -= 1.0;
        }

        // Back-propagate g_emit through the output and hidden layers.
        let mut g_w1 = vec![0.0; hh * d];
        let mut g_b1 = vec![0.0; hh];
        let mut g_w2 = vec![0.0; k * hh];
        let mut g_b2 = vec![0.0; k];

        for t in 0..t_max {
            let xt = &x[t * d..(t + 1) * d];
            let h_t = &fwd.hidden[t * hh..(t + 1) * hh];
            // Output layer: e = W2 h + b2.
            for tag in 0..k {
                let ge = g_emit[t * k + tag];
                g_b2[tag] += ge;
                let row = tag * hh;
                for h in 0..hh {
                    g_w2[row + h] += ge * h_t[h];
                }
            }
            // Gradient flowing into hidden activations, then through tanh.
            for h in 0..hh {
                let mut g_h = 0.0;
                for tag in 0..k {
                    g_h += g_emit[t * k + tag] * self.w2[tag * hh + h];
                }
                // d tanh / d pre = 1 − tanh²; here h_t[h] is tanh(pre).
                let g_pre = g_h * (1.0 - h_t[h] * h_t[h]);
                g_b1[h] += g_pre;
                let row = h * d;
                for (dd, &xv) in xt.iter().enumerate() {
                    g_w1[row + dd] += g_pre * xv;
                }
            }
        }

        Ok((
            nll,
            NeuralCrfGrad {
                w1: g_w1,
                b1: g_b1,
                w2: g_w2,
                b2: g_b2,
                transitions: g_trans,
            },
        ))
    }

    /// Apply one gradient-descent step with learning rate `lr` on a single example.
    ///
    /// Computes the forward pass, back-propagates the NLL gradient, updates every
    /// parameter in place, and returns the NLL *before* the update.
    pub fn step(&mut self, x: &[f64], y: &[usize], lr: f64) -> SeqResult<f64> {
        if !lr.is_finite() || lr <= 0.0 {
            return Err(SeqError::InvalidParameter {
                name: "lr".to_string(),
                value: lr,
            });
        }
        let fwd = self.forward(x)?;
        let (nll, grad) = self.backward(x, &fwd, y)?;
        for (w, g) in self.w1.iter_mut().zip(grad.w1.iter()) {
            *w -= lr * g;
        }
        for (w, g) in self.b1.iter_mut().zip(grad.b1.iter()) {
            *w -= lr * g;
        }
        for (w, g) in self.w2.iter_mut().zip(grad.w2.iter()) {
            *w -= lr * g;
        }
        for (w, g) in self.b2.iter_mut().zip(grad.b2.iter()) {
            *w -= lr * g;
        }
        for (w, g) in self.transitions.iter_mut().zip(grad.transitions.iter()) {
            *w -= lr * g;
        }
        Ok(nll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force log-partition: log-sum-exp of the score over all `K^T` paths.
    fn brute_log_partition(net: &NeuralCrf, emit: &[f64]) -> f64 {
        let k = net.n_tags;
        let t_max = emit.len() / k;
        let mut scores: Vec<f64> = Vec::new();
        let mut y = vec![0usize; t_max];
        loop {
            let s = net.sequence_score(emit, &y).expect("score");
            scores.push(s);
            // Odometer increment over the K^T tag grid.
            let mut pos = 0;
            loop {
                if pos == t_max {
                    return logsumexp(&scores);
                }
                y[pos] += 1;
                if y[pos] < k {
                    break;
                }
                y[pos] = 0;
                pos += 1;
            }
        }
    }

    /// Brute-force argmax path by exhaustive enumeration.
    fn brute_viterbi(net: &NeuralCrf, emit: &[f64]) -> Vec<usize> {
        let k = net.n_tags;
        let t_max = emit.len() / k;
        let mut best_y = vec![0usize; t_max];
        let mut best_s = f64::NEG_INFINITY;
        let mut y = vec![0usize; t_max];
        loop {
            let s = net.sequence_score(emit, &y).expect("score");
            if s > best_s {
                best_s = s;
                best_y = y.clone();
            }
            let mut pos = 0;
            loop {
                if pos == t_max {
                    return best_y;
                }
                y[pos] += 1;
                if y[pos] < k {
                    break;
                }
                y[pos] = 0;
                pos += 1;
            }
        }
    }

    fn toy_net() -> NeuralCrf {
        let mut rng = LcgRng::new(7);
        let mut net = NeuralCrf::new(3, 4, 5, 0.4, &mut rng).expect("net");
        for (i, v) in net.transitions.iter_mut().enumerate() {
            *v = ((i as f64) * 0.13 - 0.2).sin() * 0.3;
        }
        for v in net.b2.iter_mut() {
            *v = 0.1;
        }
        net
    }

    fn toy_features(net: &NeuralCrf, t_max: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..t_max * net.input_dim)
            .map(|_| rng.next_range(-1.0, 1.0))
            .collect()
    }

    #[test]
    fn construct_validates_dims() {
        assert!(NeuralCrf::zeros(0, 2, 2).is_err());
        assert!(NeuralCrf::zeros(2, 0, 2).is_err());
        assert!(NeuralCrf::zeros(2, 2, 0).is_err());
        let net = NeuralCrf::zeros(3, 4, 5).expect("ok");
        assert_eq!(net.param_count(), 5 * 4 + 5 + 3 * 5 + 3 + 3 * 3);
    }

    #[test]
    fn new_rejects_bad_scale() {
        let mut rng = LcgRng::new(1);
        assert!(NeuralCrf::new(2, 2, 2, 0.0, &mut rng).is_err());
        assert!(NeuralCrf::new(2, 2, 2, -1.0, &mut rng).is_err());
        assert!(NeuralCrf::new(2, 2, 2, f64::NAN, &mut rng).is_err());
    }

    #[test]
    fn forward_shapes_and_emit_match_manual() {
        let net = toy_net();
        let x = toy_features(&net, 4, 11);
        let fwd = net.forward(&x).expect("fwd");
        assert_eq!(fwd.t_max, 4);
        assert_eq!(fwd.hidden.len(), 4 * net.hidden_dim);
        assert_eq!(fwd.emit.len(), 4 * net.n_tags);
        // Recompute emission for (t=2, tag=1) by hand.
        let d = net.input_dim;
        let hh = net.hidden_dim;
        let t = 2usize;
        let tag = 1usize;
        let mut acc = net.b2[tag];
        for h in 0..hh {
            let mut pre = net.b1[h];
            for dd in 0..d {
                pre += net.w1[h * d + dd] * x[t * d + dd];
            }
            acc += net.w2[tag * hh + h] * pre.tanh();
        }
        assert!((acc - fwd.emit[t * net.n_tags + tag]).abs() < 1e-12);
    }

    #[test]
    fn log_partition_matches_brute_force() {
        let net = toy_net();
        for (seed, t_max) in [(3u64, 2usize), (5, 3), (9, 4)] {
            let x = toy_features(&net, t_max, seed);
            let fwd = net.forward(&x).expect("fwd");
            let via_forward = net.log_partition(&fwd.emit).expect("logz");
            let via_brute = brute_log_partition(&net, &fwd.emit);
            assert!(
                (via_forward - via_brute).abs() < 1e-9,
                "T={t_max}: forward={via_forward}, brute={via_brute}"
            );
        }
    }

    #[test]
    fn viterbi_matches_brute_force_argmax() {
        let net = toy_net();
        for (seed, t_max) in [(2u64, 2usize), (4, 3), (6, 4), (8, 5)] {
            let x = toy_features(&net, t_max, seed);
            let fwd = net.forward(&x).expect("fwd");
            let path = net.viterbi(&fwd.emit).expect("viterbi");
            let brute = brute_viterbi(&net, &fwd.emit);
            // Scores must coincide (path may differ only on exact ties).
            let s_path = net.sequence_score(&fwd.emit, &path).expect("s");
            let s_brute = net.sequence_score(&fwd.emit, &brute).expect("s");
            assert!((s_path - s_brute).abs() < 1e-9, "T={t_max}");
            assert_eq!(path, brute, "T={t_max}");
        }
    }

    #[test]
    fn decode_returns_in_range_path() {
        let net = toy_net();
        let x = toy_features(&net, 6, 21);
        let path = net.decode(&x).expect("decode");
        assert_eq!(path.len(), 6);
        assert!(path.iter().all(|&p| p < net.n_tags));
    }

    #[test]
    fn nll_is_nonnegative_and_consistent() {
        let net = toy_net();
        let x = toy_features(&net, 4, 31);
        let y = vec![0usize, 2, 1, 0];
        let direct = net.nll(&x, &y).expect("nll");
        let fwd = net.forward(&x).expect("fwd");
        let cached = net.nll_from_forward(&fwd, &y).expect("nll2");
        assert!((direct - cached).abs() < 1e-12);
        // NLL = log Z − score ≥ 0 since the gold score ≤ log Z.
        assert!(direct >= -1e-9, "nll={direct}");
    }

    #[test]
    fn emission_and_transition_gradients_match_finite_difference() {
        let net = toy_net();
        let x = toy_features(&net, 4, 41);
        let y = vec![1usize, 0, 2, 1];
        let fwd = net.forward(&x).expect("fwd");
        let (_, grad) = net.backward(&x, &fwd, &y).expect("bwd");

        let eps = 1e-6;
        // Helper closures perturbing a chosen parameter array.
        let central = |perturb: &dyn Fn(&mut NeuralCrf, f64)| -> f64 {
            let mut up = net.clone();
            perturb(&mut up, eps);
            let mut dn = net.clone();
            perturb(&mut dn, -eps);
            let lp = up.nll(&x, &y).expect("nll+");
            let lm = dn.nll(&x, &y).expect("nll-");
            (lp - lm) / (2.0 * eps)
        };

        for idx in 0..net.w1.len() {
            let num = central(&|n, e| n.w1[idx] += e);
            assert!(
                (num - grad.w1[idx]).abs() < 1e-4,
                "w1[{idx}] num={num} ana={}",
                grad.w1[idx]
            );
        }
        for idx in 0..net.w2.len() {
            let num = central(&|n, e| n.w2[idx] += e);
            assert!(
                (num - grad.w2[idx]).abs() < 1e-4,
                "w2[{idx}] num={num} ana={}",
                grad.w2[idx]
            );
        }
        for idx in 0..net.b1.len() {
            let num = central(&|n, e| n.b1[idx] += e);
            assert!(
                (num - grad.b1[idx]).abs() < 1e-4,
                "b1[{idx}] num={num} ana={}",
                grad.b1[idx]
            );
        }
        for idx in 0..net.b2.len() {
            let num = central(&|n, e| n.b2[idx] += e);
            assert!(
                (num - grad.b2[idx]).abs() < 1e-4,
                "b2[{idx}] num={num} ana={}",
                grad.b2[idx]
            );
        }
        for idx in 0..net.transitions.len() {
            let num = central(&|n, e| n.transitions[idx] += e);
            assert!(
                (num - grad.transitions[idx]).abs() < 1e-4,
                "trans[{idx}] num={num} ana={}",
                grad.transitions[idx]
            );
        }
    }

    #[test]
    fn training_reduces_nll_on_toy_sequence() {
        let mut net = toy_net();
        let x = toy_features(&net, 5, 51);
        let y = vec![0usize, 1, 2, 1, 0];
        let nll0 = net.nll(&x, &y).expect("nll0");
        for _ in 0..200 {
            net.step(&x, &y, 0.05).expect("step");
        }
        let nll1 = net.nll(&x, &y).expect("nll1");
        assert!(nll1 < nll0 - 1e-3, "nll0={nll0}, nll1={nll1}");
        // After training the decoded path should match the gold sequence.
        let path = net.decode(&x).expect("decode");
        assert_eq!(path, y);
    }

    #[test]
    fn step_validates_learning_rate() {
        let mut net = toy_net();
        let x = toy_features(&net, 3, 61);
        let y = vec![0usize, 1, 2];
        assert!(net.step(&x, &y, 0.0).is_err());
        assert!(net.step(&x, &y, -0.1).is_err());
    }

    #[test]
    fn input_validation_paths() {
        let net = toy_net();
        // Empty input.
        assert!(net.forward(&[]).is_err());
        // Ragged input length (not a multiple of input_dim).
        let bad = vec![0.0; net.input_dim * 2 + 1];
        assert!(net.forward(&bad).is_err());
        // Gold tag out of range.
        let x = toy_features(&net, 2, 71);
        assert!(net.nll(&x, &[0, net.n_tags]).is_err());
        // Length mismatch between y and T.
        assert!(net.nll(&x, &[0]).is_err());
    }

    #[test]
    fn marginals_form_valid_distributions() {
        let net = toy_net();
        let x = toy_features(&net, 4, 81);
        let fwd = net.forward(&x).expect("fwd");
        let (p_node, p_edge) = net.marginals(&fwd.emit).expect("marg");
        let k = net.n_tags;
        for t in 0..fwd.t_max {
            let s: f64 = p_node[t * k..t * k + k].iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "node t={t} sum={s}");
            assert!(p_node[t * k..t * k + k].iter().all(|&p| p >= -1e-12));
        }
        for t in 0..fwd.t_max - 1 {
            let s: f64 = p_edge[t * k * k..(t + 1) * k * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "edge t={t} sum={s}");
        }
    }
}
