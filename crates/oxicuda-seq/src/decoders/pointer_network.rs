//! Pointer Network.
//!
//! Reference: Vinyals, O., Fortunato, M. & Jaitly, N. (2015). *Pointer Networks*.
//! NeurIPS 28 (arXiv 1506.03134). <https://arxiv.org/abs/1506.03134>.
//!
//! # Model
//!
//! A Pointer Network is a sequence-to-sequence model whose attention mechanism
//! **points to positions in the input** rather than emitting a token from a fixed
//! output vocabulary. This makes the output vocabulary equal to the (variable)
//! input length `n`, which is exactly what combinatorial tasks such as sorting,
//! convex hull and the travelling-salesman problem require.
//!
//! Given encoder hidden states `e_1 … e_n` and a decoder query `d_i`, the
//! content-based attention score for pointing at input position `j` is
//!
//! ```text
//! u^i_j = vᵀ tanh(W1 e_j + W2 d_i)
//! ```
//!
//! and the **pointer distribution** over input positions is
//! `p^i = softmax(u^i)`. Greedy decoding emits `argmax_j p^i_j` at each step.
//!
//! Here the encoder states are provided directly (or produced by a minimal
//! Elman/`tanh` RNN encoder, [`PointerNetwork::encode`]) and the decoder queries
//! are likewise provided per step, so the module is a faithful CPU reference for
//! the pointer attention head and its training objective (teacher-forced NLL with
//! a finite-difference-verified gradient) without committing to any one recurrent
//! cell. Production code never panics: all fallible paths return [`SeqError`].

use crate::error::{SeqError, SeqResult};
use crate::handle::LcgRng;

/// A Pointer Network attention head with optional Elman encoder.
///
/// Parameter layout (all row-major, `f64`):
///
/// * `w1[a * hidden_dim + h]` — encoder projection (`attn_dim × hidden_dim`)
/// * `w2[a * hidden_dim + h]` — decoder-query projection (`attn_dim × hidden_dim`)
/// * `v[a]` — attention combination vector (`attn_dim`)
/// * `enc_wx[h * input_dim + d]` — encoder input→hidden (`hidden_dim × input_dim`)
/// * `enc_wh[h * hidden_dim + h2]` — encoder hidden→hidden (`hidden_dim × hidden_dim`)
/// * `enc_b[h]` — encoder hidden bias (`hidden_dim`)
#[derive(Debug, Clone)]
pub struct PointerNetwork {
    /// Dimensionality of encoder/decoder hidden states.
    pub hidden_dim: usize,
    /// Attention (alignment) inner dimension.
    pub attn_dim: usize,
    /// Input feature dimensionality for the optional Elman encoder.
    pub input_dim: usize,
    /// Encoder-state projection `W1` (`attn_dim × hidden_dim`).
    pub w1: Vec<f64>,
    /// Decoder-query projection `W2` (`attn_dim × hidden_dim`).
    pub w2: Vec<f64>,
    /// Attention combination vector `v` (`attn_dim`).
    pub v: Vec<f64>,
    /// Elman encoder input→hidden weight (`hidden_dim × input_dim`).
    pub enc_wx: Vec<f64>,
    /// Elman encoder hidden→hidden weight (`hidden_dim × hidden_dim`).
    pub enc_wh: Vec<f64>,
    /// Elman encoder hidden bias (`hidden_dim`).
    pub enc_b: Vec<f64>,
}

/// Gradients of the teacher-forced NLL with respect to the attention parameters.
///
/// Only the attention head (`w1`, `w2`, `v`) is differentiated here; the encoder
/// parameters are exercised by [`PointerNetwork::encode`] but treated as fixed
/// feature extractors for the gradient check.
#[derive(Debug, Clone)]
pub struct PointerGrad {
    /// Gradient w.r.t. `w1`.
    pub w1: Vec<f64>,
    /// Gradient w.r.t. `w2`.
    pub w2: Vec<f64>,
    /// Gradient w.r.t. `v`.
    pub v: Vec<f64>,
}

impl PointerNetwork {
    /// Construct a zero-initialised pointer network.
    ///
    /// All dimensions must be positive; otherwise [`SeqError::InvalidConfiguration`].
    pub fn zeros(hidden_dim: usize, attn_dim: usize, input_dim: usize) -> SeqResult<Self> {
        if hidden_dim == 0 || attn_dim == 0 || input_dim == 0 {
            return Err(SeqError::InvalidConfiguration(
                "hidden_dim, attn_dim and input_dim must all be > 0".to_string(),
            ));
        }
        Ok(Self {
            hidden_dim,
            attn_dim,
            input_dim,
            w1: vec![0.0; attn_dim * hidden_dim],
            w2: vec![0.0; attn_dim * hidden_dim],
            v: vec![0.0; attn_dim],
            enc_wx: vec![0.0; hidden_dim * input_dim],
            enc_wh: vec![0.0; hidden_dim * hidden_dim],
            enc_b: vec![0.0; hidden_dim],
        })
    }

    /// Construct a pointer network with small random weights from a seeded LCG.
    ///
    /// All weight matrices and `v` are sampled `~ U(-scale, scale)`; biases start
    /// at zero. `scale` must be finite and positive.
    pub fn new(
        hidden_dim: usize,
        attn_dim: usize,
        input_dim: usize,
        scale: f64,
        rng: &mut LcgRng,
    ) -> SeqResult<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(SeqError::InvalidParameter {
                name: "scale".to_string(),
                value: scale,
            });
        }
        let mut net = Self::zeros(hidden_dim, attn_dim, input_dim)?;
        for buf in [&mut net.w1, &mut net.w2, &mut net.enc_wx, &mut net.enc_wh] {
            for v in buf.iter_mut() {
                *v = rng.next_range(-scale, scale);
            }
        }
        for v in net.v.iter_mut() {
            *v = rng.next_range(-scale, scale);
        }
        Ok(net)
    }

    /// Number of input positions implied by an encoder-state buffer.
    fn n_positions(&self, encoder_states: &[f64]) -> SeqResult<usize> {
        if encoder_states.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if encoder_states.len() % self.hidden_dim != 0 {
            return Err(SeqError::DimensionMismatch {
                a: encoder_states.len(),
                b: self.hidden_dim,
            });
        }
        Ok(encoder_states.len() / self.hidden_dim)
    }

    /// Encode an input feature sequence with a minimal Elman (`tanh`) RNN.
    ///
    /// `inputs` is `n × input_dim` row-major; the returned buffer is
    /// `n × hidden_dim` row-major encoder states `e_1 … e_n`. Provided as a
    /// convenience; callers may instead pass their own embeddings to the
    /// attention methods directly.
    pub fn encode(&self, inputs: &[f64]) -> SeqResult<Vec<f64>> {
        if inputs.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if inputs.len() % self.input_dim != 0 {
            return Err(SeqError::DimensionMismatch {
                a: inputs.len(),
                b: self.input_dim,
            });
        }
        let n = inputs.len() / self.input_dim;
        let hh = self.hidden_dim;
        let d = self.input_dim;
        let mut states = vec![0.0; n * hh];
        let mut prev = vec![0.0; hh];
        for t in 0..n {
            let xt = &inputs[t * d..(t + 1) * d];
            for h in 0..hh {
                let mut acc = self.enc_b[h];
                let rx = h * d;
                for (dd, &xv) in xt.iter().enumerate() {
                    acc += self.enc_wx[rx + dd] * xv;
                }
                let rh = h * hh;
                for (h2, &pv) in prev.iter().enumerate() {
                    acc += self.enc_wh[rh + h2] * pv;
                }
                states[t * hh + h] = acc.tanh();
            }
            prev.copy_from_slice(&states[t * hh..(t + 1) * hh]);
        }
        Ok(states)
    }

    /// Pre-compute the encoder projections `W1 e_j` for all positions `j`.
    ///
    /// Returns an `n × attn_dim` row-major buffer reused across decoder steps.
    fn project_encoder(&self, encoder_states: &[f64]) -> SeqResult<Vec<f64>> {
        let n = self.n_positions(encoder_states)?;
        let a = self.attn_dim;
        let hh = self.hidden_dim;
        let mut proj = vec![0.0; n * a];
        for j in 0..n {
            let ej = &encoder_states[j * hh..(j + 1) * hh];
            for aa in 0..a {
                let mut acc = 0.0;
                let row = aa * hh;
                for (h, &ev) in ej.iter().enumerate() {
                    acc += self.w1[row + h] * ev;
                }
                proj[j * a + aa] = acc;
            }
        }
        Ok(proj)
    }

    /// Project a decoder query `d_i` to `W2 d_i` (length `attn_dim`).
    fn project_query(&self, query: &[f64]) -> SeqResult<Vec<f64>> {
        if query.len() != self.hidden_dim {
            return Err(SeqError::ShapeMismatch {
                expected: self.hidden_dim,
                got: query.len(),
            });
        }
        let a = self.attn_dim;
        let hh = self.hidden_dim;
        let mut q = vec![0.0; a];
        for aa in 0..a {
            let mut acc = 0.0;
            let row = aa * hh;
            for (h, &qv) in query.iter().enumerate() {
                acc += self.w2[row + h] * qv;
            }
            q[aa] = acc;
        }
        Ok(q)
    }

    /// Raw attention logits `u^i_j = vᵀ tanh(W1 e_j + W2 d_i)` for one query.
    ///
    /// `encoder_states` is `n × hidden_dim`; `query` is length `hidden_dim`.
    pub fn attention_logits(&self, encoder_states: &[f64], query: &[f64]) -> SeqResult<Vec<f64>> {
        let proj = self.project_encoder(encoder_states)?;
        let qp = self.project_query(query)?;
        let n = self.n_positions(encoder_states)?;
        let a = self.attn_dim;
        let mut logits = vec![0.0; n];
        for j in 0..n {
            let mut acc = 0.0;
            for aa in 0..a {
                acc += self.v[aa] * (proj[j * a + aa] + qp[aa]).tanh();
            }
            logits[j] = acc;
        }
        Ok(logits)
    }

    /// Numerically-stable softmax of `logits` in place into a fresh vector.
    fn softmax(logits: &[f64]) -> Vec<f64> {
        let mut max = f64::NEG_INFINITY;
        for &z in logits {
            if z > max {
                max = z;
            }
        }
        if !max.is_finite() {
            // All −inf or empty: fall back to uniform over the support.
            let n = logits.len().max(1);
            return vec![1.0 / n as f64; logits.len()];
        }
        let mut probs: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
        let s: f64 = probs.iter().sum();
        if s > 0.0 {
            for p in probs.iter_mut() {
                *p /= s;
            }
        }
        probs
    }

    /// Pointer distribution `softmax(u^i)` over the `n` input positions for one
    /// decoder query.
    pub fn pointer_distribution(
        &self,
        encoder_states: &[f64],
        query: &[f64],
    ) -> SeqResult<Vec<f64>> {
        let logits = self.attention_logits(encoder_states, query)?;
        Ok(Self::softmax(&logits))
    }

    /// Forward pass over a sequence of decoder queries.
    ///
    /// `queries` is `m × hidden_dim` row-major. Returns an `m × n` row-major
    /// matrix of pointer distributions (row `i` is the distribution for query `i`).
    pub fn forward(&self, encoder_states: &[f64], queries: &[f64]) -> SeqResult<Vec<f64>> {
        let n = self.n_positions(encoder_states)?;
        let hh = self.hidden_dim;
        if queries.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if queries.len() % hh != 0 {
            return Err(SeqError::DimensionMismatch {
                a: queries.len(),
                b: hh,
            });
        }
        let m = queries.len() / hh;
        let proj = self.project_encoder(encoder_states)?;
        let a = self.attn_dim;
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            let qp = self.project_query(&queries[i * hh..(i + 1) * hh])?;
            let mut logits = vec![0.0; n];
            for j in 0..n {
                let mut acc = 0.0;
                for aa in 0..a {
                    acc += self.v[aa] * (proj[j * a + aa] + qp[aa]).tanh();
                }
                logits[j] = acc;
            }
            let probs = Self::softmax(&logits);
            out[i * n..(i + 1) * n].copy_from_slice(&probs);
        }
        Ok(out)
    }

    /// Greedy decode: emit `argmax_j p^i_j` for each decoder query.
    ///
    /// Returns a sequence of `m` pointed input indices, each in `0..n`.
    pub fn decode(&self, encoder_states: &[f64], queries: &[f64]) -> SeqResult<Vec<usize>> {
        let n = self.n_positions(encoder_states)?;
        let probs = self.forward(encoder_states, queries)?;
        let m = probs.len() / n;
        let mut out = vec![0usize; m];
        for i in 0..m {
            let mut best = f64::NEG_INFINITY;
            let mut argmax = 0usize;
            for j in 0..n {
                let p = probs[i * n + j];
                if p > best {
                    best = p;
                    argmax = j;
                }
            }
            out[i] = argmax;
        }
        Ok(out)
    }

    /// Teacher-forced negative log-likelihood of a target index sequence.
    ///
    /// `targets[i]` is the gold input position pointed to at decoder step `i`;
    /// `NLL = − Σ_i log p^i_{targets[i]}`. Targets must be in `0..n`.
    pub fn nll(
        &self,
        encoder_states: &[f64],
        queries: &[f64],
        targets: &[usize],
    ) -> SeqResult<f64> {
        let n = self.n_positions(encoder_states)?;
        let probs = self.forward(encoder_states, queries)?;
        let m = probs.len() / n;
        if targets.len() != m {
            return Err(SeqError::LengthMismatch {
                a: targets.len(),
                b: m,
            });
        }
        let mut nll = 0.0;
        for i in 0..m {
            let tgt = targets[i];
            if tgt >= n {
                return Err(SeqError::IndexOutOfBounds { index: tgt, len: n });
            }
            let p = probs[i * n + tgt].max(1e-300);
            nll -= p.ln();
        }
        Ok(nll)
    }

    /// Gradient of the teacher-forced NLL w.r.t. the attention parameters
    /// (`w1`, `w2`, `v`).
    ///
    /// Uses the softmax-cross-entropy identity `∂NLL/∂u^i_j = p^i_j − 1[j = tgt_i]`
    /// and back-propagates through `s_{j,a} = tanh(W1 e_j + W2 d_i)_a`. Returns the
    /// NLL and a [`PointerGrad`].
    pub fn backward(
        &self,
        encoder_states: &[f64],
        queries: &[f64],
        targets: &[usize],
    ) -> SeqResult<(f64, PointerGrad)> {
        let n = self.n_positions(encoder_states)?;
        let hh = self.hidden_dim;
        if queries.is_empty() || queries.len() % hh != 0 {
            return Err(SeqError::DimensionMismatch {
                a: queries.len(),
                b: hh,
            });
        }
        let m = queries.len() / hh;
        if targets.len() != m {
            return Err(SeqError::LengthMismatch {
                a: targets.len(),
                b: m,
            });
        }
        for &t in targets {
            if t >= n {
                return Err(SeqError::IndexOutOfBounds { index: t, len: n });
            }
        }
        let a = self.attn_dim;
        let proj = self.project_encoder(encoder_states)?;

        let mut g_w1 = vec![0.0; a * hh];
        let mut g_w2 = vec![0.0; a * hh];
        let mut g_v = vec![0.0; a];
        let mut nll = 0.0;

        for i in 0..m {
            let qi = &queries[i * hh..(i + 1) * hh];
            let qp = self.project_query(qi)?;
            // Per-position pre-activation s and its tanh, plus logits.
            let mut s = vec![0.0; n * a];
            let mut logits = vec![0.0; n];
            for j in 0..n {
                let mut acc = 0.0;
                for aa in 0..a {
                    let pre = proj[j * a + aa] + qp[aa];
                    let th = pre.tanh();
                    s[j * a + aa] = th;
                    acc += self.v[aa] * th;
                }
                logits[j] = acc;
            }
            let probs = Self::softmax(&logits);
            let tgt = targets[i];
            nll -= probs[tgt].max(1e-300).ln();

            // d_logit[j] = p_j − 1[j == tgt]
            for j in 0..n {
                let d_logit = probs[j] - if j == tgt { 1.0 } else { 0.0 };
                let ej = &encoder_states[j * hh..(j + 1) * hh];
                for aa in 0..a {
                    // v gradient: u depends on v_a via s_{j,a}.
                    g_v[aa] += d_logit * s[j * a + aa];
                    // Through tanh into the pre-activation.
                    let d_pre = d_logit * self.v[aa] * (1.0 - s[j * a + aa] * s[j * a + aa]);
                    let row = aa * hh;
                    for h in 0..hh {
                        // W1 e_j contribution.
                        g_w1[row + h] += d_pre * ej[h];
                        // W2 d_i contribution.
                        g_w2[row + h] += d_pre * qi[h];
                    }
                }
            }
        }

        Ok((
            nll,
            PointerGrad {
                w1: g_w1,
                w2: g_w2,
                v: g_v,
            },
        ))
    }

    /// Apply one gradient-descent step on the attention parameters with learning
    /// rate `lr`. Returns the NLL *before* the update.
    pub fn step(
        &mut self,
        encoder_states: &[f64],
        queries: &[f64],
        targets: &[usize],
        lr: f64,
    ) -> SeqResult<f64> {
        if !lr.is_finite() || lr <= 0.0 {
            return Err(SeqError::InvalidParameter {
                name: "lr".to_string(),
                value: lr,
            });
        }
        let (nll, grad) = self.backward(encoder_states, queries, targets)?;
        for (w, g) in self.w1.iter_mut().zip(grad.w1.iter()) {
            *w -= lr * g;
        }
        for (w, g) in self.w2.iter_mut().zip(grad.w2.iter()) {
            *w -= lr * g;
        }
        for (w, g) in self.v.iter_mut().zip(grad.v.iter()) {
            *w -= lr * g;
        }
        Ok(nll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_net(seed: u64) -> PointerNetwork {
        let mut rng = LcgRng::new(seed);
        PointerNetwork::new(3, 4, 2, 0.5, &mut rng).expect("net")
    }

    fn rand_states(net: &PointerNetwork, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n * net.hidden_dim)
            .map(|_| rng.next_range(-1.0, 1.0))
            .collect()
    }

    fn rand_queries(net: &PointerNetwork, m: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..m * net.hidden_dim)
            .map(|_| rng.next_range(-1.0, 1.0))
            .collect()
    }

    #[test]
    fn construct_validates_dims() {
        assert!(PointerNetwork::zeros(0, 2, 2).is_err());
        assert!(PointerNetwork::zeros(2, 0, 2).is_err());
        assert!(PointerNetwork::zeros(2, 2, 0).is_err());
        let mut rng = LcgRng::new(1);
        assert!(PointerNetwork::new(2, 2, 2, 0.0, &mut rng).is_err());
        assert!(PointerNetwork::new(2, 2, 2, f64::INFINITY, &mut rng).is_err());
    }

    #[test]
    fn pointer_distribution_is_valid_simplex() {
        let net = rand_net(2);
        let states = rand_states(&net, 5, 3);
        let query = rand_queries(&net, 1, 4);
        let dist = net.pointer_distribution(&states, &query).expect("dist");
        assert_eq!(dist.len(), 5);
        assert!(dist.iter().all(|&p| (0.0..=1.0).contains(&p)));
        let s: f64 = dist.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "sum={s}");
    }

    #[test]
    fn attention_shapes_correct() {
        let net = rand_net(5);
        let n = 6usize;
        let m = 4usize;
        let states = rand_states(&net, n, 6);
        let queries = rand_queries(&net, m, 7);
        let logits = net
            .attention_logits(&states, &queries[..net.hidden_dim])
            .expect("logits");
        assert_eq!(logits.len(), n);
        let probs = net.forward(&states, &queries).expect("fwd");
        assert_eq!(probs.len(), m * n);
        // Each row is a simplex.
        for i in 0..m {
            let s: f64 = probs[i * n..(i + 1) * n].iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "row {i} sum={s}");
        }
    }

    #[test]
    fn decode_yields_in_range_indices() {
        let net = rand_net(8);
        let n = 7usize;
        let states = rand_states(&net, n, 9);
        let queries = rand_queries(&net, 5, 10);
        let path = net.decode(&states, &queries).expect("decode");
        assert_eq!(path.len(), 5);
        assert!(path.iter().all(|&p| p < n));
    }

    #[test]
    fn decode_is_deterministic() {
        let net = rand_net(11);
        let states = rand_states(&net, 6, 12);
        let queries = rand_queries(&net, 4, 13);
        let p1 = net.decode(&states, &queries).expect("d1");
        let p2 = net.decode(&states, &queries).expect("d2");
        assert_eq!(p1, p2);
        let f1 = net.forward(&states, &queries).expect("f1");
        let f2 = net.forward(&states, &queries).expect("f2");
        assert_eq!(f1, f2);
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let net = rand_net(14);
        let n = 5usize;
        let states = rand_states(&net, n, 15);
        let queries = rand_queries(&net, 3, 16);
        let targets = vec![2usize, 0, 4];
        let (_, grad) = net.backward(&states, &queries, &targets).expect("bwd");

        let eps = 1e-6;
        let central = |perturb: &dyn Fn(&mut PointerNetwork, f64)| -> f64 {
            let mut up = net.clone();
            perturb(&mut up, eps);
            let mut dn = net.clone();
            perturb(&mut dn, -eps);
            let lp = up.nll(&states, &queries, &targets).expect("nll+");
            let lm = dn.nll(&states, &queries, &targets).expect("nll-");
            (lp - lm) / (2.0 * eps)
        };

        for idx in 0..net.w1.len() {
            let num = central(&|p, e| p.w1[idx] += e);
            assert!(
                (num - grad.w1[idx]).abs() < 1e-4,
                "w1[{idx}] num={num} ana={}",
                grad.w1[idx]
            );
        }
        for idx in 0..net.w2.len() {
            let num = central(&|p, e| p.w2[idx] += e);
            assert!(
                (num - grad.w2[idx]).abs() < 1e-4,
                "w2[{idx}] num={num} ana={}",
                grad.w2[idx]
            );
        }
        for idx in 0..net.v.len() {
            let num = central(&|p, e| p.v[idx] += e);
            assert!(
                (num - grad.v[idx]).abs() < 1e-4,
                "v[{idx}] num={num} ana={}",
                grad.v[idx]
            );
        }
    }

    #[test]
    fn constructed_weights_point_to_argmax_by_key() {
        // Selection task: each encoder state e_j = [key_j, 0]. With v = [1, 0],
        // W1 = [[c, 0], [0, 0]], W2 = 0, the logit u_j = tanh(c · key_j), which is
        // monotone in key_j, so argmax points at the position with the largest key.
        let mut net = PointerNetwork::zeros(2, 2, 2).expect("net");
        let c = 3.0;
        net.w1[0] = c; // attn row 0 reads hidden dim 0 (the key)
        net.v[0] = 1.0;
        let keys = [0.2_f64, 0.9, 0.1, 0.5];
        let n = keys.len();
        let mut states = vec![0.0; n * net.hidden_dim];
        for (j, &kk) in keys.iter().enumerate() {
            states[j * net.hidden_dim] = kk;
        }
        let query = vec![0.0; net.hidden_dim];
        let dist = net.pointer_distribution(&states, &query).expect("dist");
        // argmax key is position 1.
        let argmax = dist
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("cmp"))
            .map(|(j, _)| j)
            .expect("nonempty");
        assert_eq!(argmax, 1);
    }

    #[test]
    fn training_reduces_nll_on_selection_task() {
        // Teach the net to point at the highest-key position. The decoder query is
        // a constant; the encoder states carry distinct keys in hidden dim 0.
        // A 2-dim hidden state makes the keys easy to set explicitly.
        let mut rng = LcgRng::new(21);
        let mut net = PointerNetwork::new(2, 3, 2, 0.3, &mut rng).expect("net");
        let keys = [0.1_f64, 0.4, 0.95, 0.2, 0.6];
        let n = keys.len();
        let mut states = vec![0.0; n * net.hidden_dim];
        for (j, &kk) in keys.iter().enumerate() {
            states[j * net.hidden_dim] = kk;
            states[j * net.hidden_dim + 1] = 1.0; // bias-like constant feature
        }
        let queries = vec![1.0; net.hidden_dim]; // single decoder step
        let targets = vec![2usize]; // argmax key is position 2
        let nll0 = net.nll(&states, &queries, &targets).expect("nll0");
        for _ in 0..400 {
            net.step(&states, &queries, &targets, 0.2).expect("step");
        }
        let nll1 = net.nll(&states, &queries, &targets).expect("nll1");
        assert!(nll1 < nll0 - 1e-3, "nll0={nll0}, nll1={nll1}");
        let path = net.decode(&states, &queries).expect("decode");
        assert_eq!(path, targets);
    }

    #[test]
    fn nll_validates_targets() {
        let net = rand_net(30);
        let n = 4usize;
        let states = rand_states(&net, n, 31);
        let queries = rand_queries(&net, 2, 32);
        // Target out of range.
        assert!(net.nll(&states, &queries, &[0, n]).is_err());
        // Wrong number of targets.
        assert!(net.nll(&states, &queries, &[0]).is_err());
    }

    #[test]
    fn input_validation_paths() {
        let net = rand_net(40);
        // Empty encoder states.
        assert!(
            net.pointer_distribution(&[], &vec![0.0; net.hidden_dim])
                .is_err()
        );
        // Ragged encoder states.
        let bad = vec![0.0; net.hidden_dim * 2 + 1];
        assert!(
            net.attention_logits(&bad, &vec![0.0; net.hidden_dim])
                .is_err()
        );
        // Query of wrong length.
        let states = rand_states(&net, 3, 41);
        assert!(net.pointer_distribution(&states, &[0.0, 0.0]).is_err());
        // Empty queries.
        assert!(net.forward(&states, &[]).is_err());
    }

    #[test]
    fn encoder_runs_and_shapes_match() {
        let net = rand_net(50);
        let n = 4usize;
        let inputs: Vec<f64> = {
            let mut rng = LcgRng::new(51);
            (0..n * net.input_dim)
                .map(|_| rng.next_range(-1.0, 1.0))
                .collect()
        };
        let states = net.encode(&inputs).expect("encode");
        assert_eq!(states.len(), n * net.hidden_dim);
        assert!(states.iter().all(|v| v.is_finite()));
        // States feed straight into the attention head.
        let query = vec![0.5; net.hidden_dim];
        let dist = net.pointer_distribution(&states, &query).expect("dist");
        let s: f64 = dist.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn step_validates_learning_rate() {
        let mut net = rand_net(60);
        let states = rand_states(&net, 3, 61);
        let queries = rand_queries(&net, 2, 62);
        let targets = vec![0usize, 1];
        assert!(net.step(&states, &queries, &targets, 0.0).is_err());
        assert!(net.step(&states, &queries, &targets, -1.0).is_err());
    }
}
