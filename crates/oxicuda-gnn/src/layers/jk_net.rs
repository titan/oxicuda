//! Jumping Knowledge Network (JK-Net) representation aggregator.
//!
//! Reference: Xu, Li, Tian, Sonobe, Kawarabayashi & Jegelka,
//! *"Representation Learning on Graphs with Jumping Knowledge Networks"*,
//! ICML 2018.
//!
//! A JK-Net combines the per-node representations produced at *every* GNN layer
//! `{h^(1), …, h^(L)}` into a single final per-node representation, letting the
//! model adaptively select an effective neighbourhood range for each node.
//! Three combination strategies are provided:
//!
//! - [`JkMode::Concat`]: concatenate the `L` per-layer vectors.
//! - [`JkMode::MaxPool`]: element-wise maximum across the `L` layers.
//! - [`JkMode::LstmAttention`]: run an LSTM over the layer sequence and use a
//!   learned attention score over the hidden states to form a weighted sum of
//!   the layer representations.
//!
//! # LSTM attention variant
//!
//! The original paper uses a **bi-directional** LSTM whose forward and backward
//! hidden states at each layer are concatenated before scoring. To keep the
//! implementation self-contained and deterministic, this module implements the
//! **forward-LSTM attention** variant: a single left-to-right LSTM produces
//! hidden states `{s^(1), …, s^(L)}`, each layer is scored by
//! `score_vec · s^(l)`, the scores are normalised with a softmax over the `L`
//! layers, and the output is `Σ_l α_l · h^(l)`.

use crate::error::{GnnError, GnnResult};
use crate::handle::LcgRng;

/// Combination strategy for [`JkNet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JkMode {
    /// Concatenate all layer representations (`output_dim = n_layers * dim`).
    Concat,
    /// Element-wise maximum across layers (`output_dim = dim`).
    MaxPool,
    /// Forward-LSTM attention weighted sum (`output_dim = dim`).
    LstmAttention,
}

/// Configuration for a [`JkNet`] aggregator.
#[derive(Debug, Clone)]
pub struct JkNetConfig {
    /// Number of layer representations to combine.
    pub n_layers: usize,
    /// Feature dimension of each layer representation.
    pub dim: usize,
    /// Combination mode.
    pub mode: JkMode,
    /// LSTM hidden size (only used by [`JkMode::LstmAttention`]).
    pub lstm_hidden: usize,
}

/// Parameters of the forward LSTM cell plus the attention score vector.
///
/// All gate input matrices are `lstm_hidden × dim` row-major, the recurrent
/// matrices are `lstm_hidden × lstm_hidden` row-major, and biases have length
/// `lstm_hidden`.
struct LstmParams {
    w_i: Vec<f32>,
    w_f: Vec<f32>,
    w_g: Vec<f32>,
    w_o: Vec<f32>,
    u_i: Vec<f32>,
    u_f: Vec<f32>,
    u_g: Vec<f32>,
    u_o: Vec<f32>,
    b_i: Vec<f32>,
    b_f: Vec<f32>,
    b_g: Vec<f32>,
    b_o: Vec<f32>,
    /// Attention score vector, length `lstm_hidden`.
    score: Vec<f32>,
}

/// A Jumping Knowledge aggregator.
pub struct JkNet {
    config: JkNetConfig,
    /// LSTM parameters; present only for [`JkMode::LstmAttention`].
    lstm: Option<LstmParams>,
}

impl JkNet {
    /// Construct a JK-Net aggregator.
    ///
    /// For [`JkMode::LstmAttention`] the LSTM weights are initialised from a
    /// small unit-normal distribution scaled by `sqrt(1 / dim)` (input) and
    /// `sqrt(1 / lstm_hidden)` (recurrent / score). Biases are zero except the
    /// forget-gate bias, which is set to `1.0` (a standard trick that keeps the
    /// cell state flowing early in training). For the other modes no parameters
    /// are needed.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::InvalidLayerConfig`] if `n_layers == 0`, `dim == 0`,
    /// or (for [`JkMode::LstmAttention`]) `lstm_hidden == 0`.
    pub fn new(config: JkNetConfig, rng: &mut LcgRng) -> GnnResult<Self> {
        if config.n_layers == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "JK-Net: n_layers must be > 0".to_string(),
            ));
        }
        if config.dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "JK-Net: dim must be > 0".to_string(),
            ));
        }

        let lstm = if config.mode == JkMode::LstmAttention {
            if config.lstm_hidden == 0 {
                return Err(GnnError::InvalidLayerConfig(
                    "JK-Net: lstm_hidden must be > 0 for LstmAttention".to_string(),
                ));
            }
            let dim = config.dim;
            let hidden = config.lstm_hidden;
            let in_scale = (1.0_f32 / dim as f32).sqrt();
            let rec_scale = (1.0_f32 / hidden as f32).sqrt();

            // Draw weights in a fixed order (gates: i, f, g, o; then input
            // matrices before recurrent matrices) so initialisation is fully
            // deterministic for a given seed.
            let w_i = sample_normal(hidden * dim, in_scale, rng);
            let w_f = sample_normal(hidden * dim, in_scale, rng);
            let w_g = sample_normal(hidden * dim, in_scale, rng);
            let w_o = sample_normal(hidden * dim, in_scale, rng);
            let u_i = sample_normal(hidden * hidden, rec_scale, rng);
            let u_f = sample_normal(hidden * hidden, rec_scale, rng);
            let u_g = sample_normal(hidden * hidden, rec_scale, rng);
            let u_o = sample_normal(hidden * hidden, rec_scale, rng);
            let score = sample_normal(hidden, rec_scale, rng);
            Some(LstmParams {
                w_i,
                w_f,
                w_g,
                w_o,
                u_i,
                u_f,
                u_g,
                u_o,
                b_i: vec![0.0_f32; hidden],
                // Forget-gate bias initialised to 1.0.
                b_f: vec![1.0_f32; hidden],
                b_g: vec![0.0_f32; hidden],
                b_o: vec![0.0_f32; hidden],
                score,
            })
        } else {
            None
        };

        Ok(Self { config, lstm })
    }

    /// Combine per-layer representations into the final per-node representation.
    ///
    /// # Arguments
    ///
    /// - `layer_reps`: exactly `n_layers` entries, each an `[n_nodes × dim]`
    ///   row-major matrix.
    /// - `n_nodes`: number of nodes.
    ///
    /// # Returns
    ///
    /// `[n_nodes × output_dim]` row-major, where `output_dim` is given by
    /// [`Self::output_dim`].
    ///
    /// # Errors
    ///
    /// Returns an error if `layer_reps.len() != n_layers` or if any layer matrix
    /// does not have length `n_nodes * dim`.
    pub fn aggregate(&self, layer_reps: &[Vec<f32>], n_nodes: usize) -> GnnResult<Vec<f32>> {
        let n_layers = self.config.n_layers;
        let dim = self.config.dim;

        if layer_reps.len() != n_layers {
            return Err(GnnError::DimensionMismatch {
                expected: n_layers,
                got: layer_reps.len(),
            });
        }
        for rep in layer_reps {
            if rep.len() != n_nodes * dim {
                return Err(GnnError::DimensionMismatch {
                    expected: n_nodes * dim,
                    got: rep.len(),
                });
            }
        }

        match self.config.mode {
            JkMode::Concat => Ok(self.aggregate_concat(layer_reps, n_nodes)),
            JkMode::MaxPool => Ok(self.aggregate_maxpool(layer_reps, n_nodes)),
            JkMode::LstmAttention => self.aggregate_lstm(layer_reps, n_nodes),
        }
    }

    /// Output feature dimension.
    ///
    /// `Concat` yields `n_layers * dim`; `MaxPool` and `LstmAttention` yield `dim`.
    #[inline]
    pub fn output_dim(&self) -> usize {
        match self.config.mode {
            JkMode::Concat => self.config.n_layers * self.config.dim,
            JkMode::MaxPool | JkMode::LstmAttention => self.config.dim,
        }
    }

    // ─── Concat ────────────────────────────────────────────────────────────

    fn aggregate_concat(&self, layer_reps: &[Vec<f32>], n_nodes: usize) -> Vec<f32> {
        let n_layers = self.config.n_layers;
        let dim = self.config.dim;
        let out_dim = n_layers * dim;
        let mut out = vec![0.0_f32; n_nodes * out_dim];
        for node in 0..n_nodes {
            for (l, rep) in layer_reps.iter().enumerate() {
                let src = &rep[node * dim..(node + 1) * dim];
                let dst_start = node * out_dim + l * dim;
                out[dst_start..dst_start + dim].copy_from_slice(src);
            }
        }
        out
    }

    // ─── MaxPool ───────────────────────────────────────────────────────────

    fn aggregate_maxpool(&self, layer_reps: &[Vec<f32>], n_nodes: usize) -> Vec<f32> {
        let dim = self.config.dim;
        let mut out = vec![f32::NEG_INFINITY; n_nodes * dim];
        for rep in layer_reps {
            for node in 0..n_nodes {
                for k in 0..dim {
                    let idx = node * dim + k;
                    let v = rep[idx];
                    if v > out[idx] {
                        out[idx] = v;
                    }
                }
            }
        }
        out
    }

    // ─── LSTM attention ──────────────────────────────────────────────────────

    fn aggregate_lstm(&self, layer_reps: &[Vec<f32>], n_nodes: usize) -> GnnResult<Vec<f32>> {
        let params = match &self.lstm {
            Some(p) => p,
            None => {
                return Err(GnnError::Internal(
                    "JK-Net: LSTM parameters missing for LstmAttention mode".to_string(),
                ));
            }
        };
        let n_layers = self.config.n_layers;
        let dim = self.config.dim;
        let hidden = self.config.lstm_hidden;

        let mut out = vec![0.0_f32; n_nodes * dim];

        // Reusable scratch buffers (one node at a time).
        let mut h_state = vec![0.0_f32; hidden];
        let mut c_state = vec![0.0_f32; hidden];
        let mut h_prev = vec![0.0_f32; hidden];
        let mut hidden_seq = vec![0.0_f32; n_layers * hidden];
        let mut scores = vec![0.0_f32; n_layers];

        for node in 0..n_nodes {
            // Reset recurrent state for this node.
            h_state.iter_mut().for_each(|v| *v = 0.0);
            c_state.iter_mut().for_each(|v| *v = 0.0);

            // Run the LSTM over the L layer representations.
            for l in 0..n_layers {
                let x = &layer_reps[l][node * dim..(node + 1) * dim];
                lstm_step(
                    params,
                    x,
                    &mut h_state,
                    &mut c_state,
                    &mut h_prev,
                    hidden,
                    dim,
                );
                hidden_seq[l * hidden..(l + 1) * hidden].copy_from_slice(&h_state);
            }

            // Attention scores: α_l ∝ exp(score · s^(l)), softmax over layers.
            for (l, score) in scores.iter_mut().enumerate() {
                let s = &hidden_seq[l * hidden..(l + 1) * hidden];
                let mut dot = 0.0_f32;
                for (&sc, &sv) in params.score.iter().zip(s.iter()) {
                    dot += sc * sv;
                }
                *score = dot;
            }
            softmax_in_place(&mut scores);

            // Weighted sum of the (original) layer representations.
            let dst = &mut out[node * dim..(node + 1) * dim];
            for (l, &alpha) in scores.iter().enumerate() {
                let rep = &layer_reps[l][node * dim..(node + 1) * dim];
                for (d, &val) in dst.iter_mut().zip(rep.iter()) {
                    *d += alpha * val;
                }
            }
        }

        Ok(out)
    }
}

/// One LSTM cell step. Updates `h_state` and `c_state` in place given input `x`.
///
/// Standard gating: `i = σ(W_i x + U_i h + b_i)`, `f = σ(...)`, `g = tanh(...)`,
/// `o = σ(...)`, `c' = f ⊙ c + i ⊙ g`, `h' = o ⊙ tanh(c')`.
///
/// The previous hidden state is snapshotted into `h_prev` so that every unit's
/// recurrent term reads the *pre-update* hidden vector, never a partially
/// overwritten one.
fn lstm_step(
    params: &LstmParams,
    x: &[f32],
    h_state: &mut [f32],
    c_state: &mut [f32],
    h_prev: &mut [f32],
    hidden: usize,
    dim: usize,
) {
    h_prev.copy_from_slice(h_state);
    for unit in 0..hidden {
        let pre_i = gate_pre(
            &params.w_i,
            &params.u_i,
            params.b_i[unit],
            x,
            h_prev,
            unit,
            dim,
            hidden,
        );
        let pre_f = gate_pre(
            &params.w_f,
            &params.u_f,
            params.b_f[unit],
            x,
            h_prev,
            unit,
            dim,
            hidden,
        );
        let pre_g = gate_pre(
            &params.w_g,
            &params.u_g,
            params.b_g[unit],
            x,
            h_prev,
            unit,
            dim,
            hidden,
        );
        let pre_o = gate_pre(
            &params.w_o,
            &params.u_o,
            params.b_o[unit],
            x,
            h_prev,
            unit,
            dim,
            hidden,
        );

        let i = sigmoid(pre_i);
        let f = sigmoid(pre_f);
        let g = pre_g.tanh();
        let o = sigmoid(pre_o);

        let c_new = f * c_state[unit] + i * g;
        c_state[unit] = c_new;
        h_state[unit] = o * c_new.tanh();
    }
}

/// Pre-activation for one gate unit: `W[unit,:]·x + U[unit,:]·h_prev + bias`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn gate_pre(
    w: &[f32],
    u: &[f32],
    bias: f32,
    x: &[f32],
    h_prev: &[f32],
    unit: usize,
    dim: usize,
    hidden: usize,
) -> f32 {
    let mut acc = bias;
    let w_row = &w[unit * dim..(unit + 1) * dim];
    for (&w_elem, &x_elem) in w_row.iter().zip(x.iter()) {
        acc += w_elem * x_elem;
    }
    let u_row = &u[unit * hidden..(unit + 1) * hidden];
    for (&u_elem, &h_elem) in u_row.iter().zip(h_prev.iter()) {
        acc += u_elem * h_elem;
    }
    acc
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable softmax in place over a slice.
fn softmax_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in values.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0_f32;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in values.iter_mut() {
            *v *= inv;
        }
    }
}

/// Draw `n` samples from `N(0, scale²)` using the RNG's normal generator.
fn sample_normal(n: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    while out.len() + 1 < n {
        let (a, b) = rng.next_normal_pair();
        out.push(a * scale);
        out.push(b * scale);
    }
    if out.len() < n {
        let (a, _) = rng.next_normal_pair();
        out.push(a * scale);
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make(mode: JkMode, n_layers: usize, dim: usize, lstm_hidden: usize, seed: u64) -> JkNet {
        let mut rng = LcgRng::new(seed);
        JkNet::new(
            JkNetConfig {
                n_layers,
                dim,
                mode,
                lstm_hidden,
            },
            &mut rng,
        )
        .expect("test invariant: aggregator must construct")
    }

    #[test]
    fn output_dim_concat() {
        let jk = make(JkMode::Concat, 3, 4, 0, 1);
        assert_eq!(jk.output_dim(), 3 * 4);
    }

    #[test]
    fn output_dim_maxpool() {
        let jk = make(JkMode::MaxPool, 3, 4, 0, 1);
        assert_eq!(jk.output_dim(), 4);
    }

    #[test]
    fn output_dim_lstm() {
        let jk = make(JkMode::LstmAttention, 3, 4, 8, 1);
        assert_eq!(jk.output_dim(), 4);
    }

    #[test]
    fn concat_length_and_values() {
        // 1 node, dim 2, 2 layers: [10, 20], [30, 40] → [10,20,30,40].
        let jk = make(JkMode::Concat, 2, 2, 0, 2);
        let layers = vec![vec![10.0_f32, 20.0], vec![30.0_f32, 40.0]];
        let out = jk
            .aggregate(&layers, 1)
            .expect("test invariant: aggregate must succeed");
        assert_eq!(out.len(), 4);
        assert_eq!(out, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn concat_multi_node() {
        // 2 nodes, dim 2, 2 layers.
        // layer0: node0=[1,2], node1=[3,4]; layer1: node0=[5,6], node1=[7,8]
        let jk = make(JkMode::Concat, 2, 2, 0, 3);
        let layers = vec![vec![1.0_f32, 2.0, 3.0, 4.0], vec![5.0_f32, 6.0, 7.0, 8.0]];
        let out = jk
            .aggregate(&layers, 2)
            .expect("test invariant: aggregate must succeed");
        // node0 = [1,2,5,6], node1 = [3,4,7,8]
        assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn maxpool_elementwise() {
        // 1 node, dim 2, layers [[1,3],[2,1]] → [2,3].
        let jk = make(JkMode::MaxPool, 2, 2, 0, 4);
        let layers = vec![vec![1.0_f32, 3.0], vec![2.0_f32, 1.0]];
        let out = jk
            .aggregate(&layers, 1)
            .expect("test invariant: aggregate must succeed");
        assert_eq!(out, vec![2.0, 3.0]);
    }

    #[test]
    fn maxpool_single_layer_identity() {
        let jk = make(JkMode::MaxPool, 1, 3, 0, 5);
        let layers = vec![vec![-1.0_f32, 2.0, 0.5]];
        let out = jk
            .aggregate(&layers, 1)
            .expect("test invariant: aggregate must succeed");
        assert_eq!(out, vec![-1.0, 2.0, 0.5]);
    }

    #[test]
    fn concat_single_layer_identity() {
        let jk = make(JkMode::Concat, 1, 3, 0, 6);
        let layers = vec![vec![7.0_f32, -2.0, 4.0]];
        let out = jk
            .aggregate(&layers, 1)
            .expect("test invariant: aggregate must succeed");
        assert_eq!(out, vec![7.0, -2.0, 4.0]);
    }

    #[test]
    fn maxpool_multi_node() {
        // 2 nodes, dim 2, 2 layers.
        let jk = make(JkMode::MaxPool, 2, 2, 0, 7);
        let layers = vec![vec![1.0_f32, 9.0, 5.0, 2.0], vec![4.0_f32, 3.0, 1.0, 8.0]];
        let out = jk
            .aggregate(&layers, 2)
            .expect("test invariant: aggregate must succeed");
        // node0 = max([1,9],[4,3]) = [4,9]; node1 = max([5,2],[1,8]) = [5,8]
        assert_eq!(out, vec![4.0, 9.0, 5.0, 8.0]);
    }

    #[test]
    fn lstm_output_length() {
        let jk = make(JkMode::LstmAttention, 3, 4, 6, 8);
        let layers = vec![vec![0.1_f32; 2 * 4]; 3];
        let out = jk
            .aggregate(&layers, 2)
            .expect("test invariant: aggregate must succeed");
        assert_eq!(out.len(), 2 * 4);
    }

    #[test]
    fn lstm_identical_reps_returns_that_rep() {
        // If all layer reps are identical, the weighted average (regardless of
        // the attention weights, which sum to 1) must equal that rep.
        let jk = make(JkMode::LstmAttention, 4, 3, 5, 9);
        let single = vec![0.5_f32, -1.5, 2.25, 0.5, -1.5, 2.25]; // 2 nodes × dim 3
        let layers = vec![single.clone(); 4];
        let out = jk
            .aggregate(&layers, 2)
            .expect("test invariant: aggregate must succeed");
        for (o, e) in out.iter().zip(single.iter()) {
            assert!((o - e).abs() < 1e-5, "{o} vs {e}");
        }
    }

    #[test]
    fn lstm_deterministic_given_seed() {
        let jk_a = make(JkMode::LstmAttention, 3, 4, 7, 4242);
        let jk_b = make(JkMode::LstmAttention, 3, 4, 7, 4242);
        let layers: Vec<Vec<f32>> = (0..3)
            .map(|l| (0..2 * 4).map(|i| (i + l) as f32 * 0.13).collect())
            .collect();
        let out_a = jk_a
            .aggregate(&layers, 2)
            .expect("test invariant: aggregate must succeed");
        let out_b = jk_b
            .aggregate(&layers, 2)
            .expect("test invariant: aggregate must succeed");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn lstm_finite_output() {
        let jk = make(JkMode::LstmAttention, 5, 6, 8, 11);
        let layers: Vec<Vec<f32>> = (0..5)
            .map(|l| (0..3 * 6).map(|i| ((i + l) as f32 - 10.0) * 0.7).collect())
            .collect();
        let out = jk
            .aggregate(&layers, 3)
            .expect("test invariant: aggregate must succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lstm_attention_weights_form_convex_combination() {
        // Output must lie within the element-wise [min, max] of the layer reps
        // since it is a convex combination (softmax weights sum to 1).
        let jk = make(JkMode::LstmAttention, 3, 1, 4, 13);
        // 1 node, dim 1, layers 2.0, 5.0, 9.0 → output in [2, 9].
        let layers = vec![vec![2.0_f32], vec![5.0_f32], vec![9.0_f32]];
        let out = jk
            .aggregate(&layers, 1)
            .expect("test invariant: aggregate must succeed");
        assert!(out[0] >= 2.0 - 1e-5 && out[0] <= 9.0 + 1e-5, "{}", out[0]);
    }

    #[test]
    fn multi_node_independent_concat() {
        // Two nodes processed independently with concat.
        let jk = make(JkMode::Concat, 2, 1, 0, 17);
        let layers = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let out = jk
            .aggregate(&layers, 2)
            .expect("test invariant: aggregate must succeed");
        // node0 = [1,3], node1 = [2,4]
        assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn err_n_layers_zero() {
        let mut rng = LcgRng::new(1);
        let res = JkNet::new(
            JkNetConfig {
                n_layers: 0,
                dim: 4,
                mode: JkMode::Concat,
                lstm_hidden: 0,
            },
            &mut rng,
        );
        assert!(matches!(res, Err(GnnError::InvalidLayerConfig(_))));
    }

    #[test]
    fn err_dim_zero() {
        let mut rng = LcgRng::new(1);
        let res = JkNet::new(
            JkNetConfig {
                n_layers: 3,
                dim: 0,
                mode: JkMode::Concat,
                lstm_hidden: 0,
            },
            &mut rng,
        );
        assert!(matches!(res, Err(GnnError::InvalidLayerConfig(_))));
    }

    #[test]
    fn err_lstm_hidden_zero() {
        let mut rng = LcgRng::new(1);
        let res = JkNet::new(
            JkNetConfig {
                n_layers: 3,
                dim: 4,
                mode: JkMode::LstmAttention,
                lstm_hidden: 0,
            },
            &mut rng,
        );
        assert!(matches!(res, Err(GnnError::InvalidLayerConfig(_))));
    }

    #[test]
    fn err_wrong_number_of_layers() {
        let jk = make(JkMode::Concat, 3, 2, 0, 19);
        let layers = vec![vec![0.0_f32; 2]; 2]; // only 2 layers, expected 3
        let res = jk.aggregate(&layers, 1);
        assert!(matches!(res, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_layer_wrong_length() {
        let jk = make(JkMode::MaxPool, 2, 3, 0, 23);
        // n_nodes = 2, dim = 3 → each layer must be length 6.
        let layers = vec![vec![0.0_f32; 6], vec![0.0_f32; 5]];
        let res = jk.aggregate(&layers, 2);
        assert!(matches!(res, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn maxpool_deterministic() {
        // Parameter-free modes are trivially deterministic.
        let jk_a = make(JkMode::MaxPool, 2, 3, 0, 100);
        let jk_b = make(JkMode::MaxPool, 2, 3, 0, 200);
        let layers = vec![vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; 2];
        let out_a = jk_a
            .aggregate(&layers, 2)
            .expect("test invariant: aggregate must succeed");
        let out_b = jk_b
            .aggregate(&layers, 2)
            .expect("test invariant: aggregate must succeed");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn forget_gate_bias_is_one() {
        let jk = make(JkMode::LstmAttention, 2, 3, 4, 55);
        let params = jk.lstm.as_ref().expect("test invariant: lstm present");
        assert!(params.b_f.iter().all(|&b| (b - 1.0).abs() < 1e-9));
        assert!(params.b_i.iter().all(|&b| b.abs() < 1e-9));
    }
}
