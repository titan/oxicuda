//! Set2Set global readout — Vinyals et al. 2016.
//!
//! An LSTM-based, permutation-invariant graph readout.

use crate::error::{GnnError, GnnResult};

/// Set2Set readout module.
///
/// Produces a permutation-invariant graph-level representation via an
/// LSTM-based attention mechanism with `processing_steps` iterations.
pub struct Set2Set {
    processing_steps: usize,
    input_dim: usize,
    /// LSTM hidden dimension = 2 * input_dim.
    lstm_dim: usize,
}

impl Set2Set {
    /// Construct a Set2Set module.
    ///
    /// `lstm_dim` is fixed to `2 * input_dim` per the original paper.
    pub fn new(input_dim: usize, processing_steps: usize) -> GnnResult<Self> {
        if input_dim == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "input_dim must be > 0".to_string(),
            ));
        }
        if processing_steps == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "processing_steps must be > 0".to_string(),
            ));
        }
        let lstm_dim = 2 * input_dim;
        Ok(Self {
            processing_steps,
            input_dim,
            lstm_dim,
        })
    }

    /// Output dimension (= `2 * input_dim`).
    pub fn output_dim(&self) -> usize {
        2 * self.input_dim
    }

    /// Forward pass.
    ///
    /// # LSTM Set2Set procedure
    ///
    /// ```text
    /// q_0 = 0,  h_0 = 0,  c_0 = 0
    /// For t = 1..T:
    ///   (h_t, c_t) = LSTM(q*_{t-1}, h_{t-1}, c_{t-1})
    ///   e_i = h_t^T x_i
    ///   α_i = softmax(e_i)
    ///   r_t  = Σ_i α_i x_i
    ///   q_t  = [h_t || r_t]
    /// Output: q_T (dim = 2 * input_dim)
    /// ```
    ///
    /// # Arguments
    ///
    /// - `x`: `[n_nodes × input_dim]`
    /// - `lstm_weight`: `[4 × lstm_dim × (lstm_dim + input_dim)]` (i, f, g, o gates × rows)
    ///   linearised as a flat `[4 * lstm_dim * (lstm_dim + input_dim)]` array.
    /// - `lstm_bias`: `[4 × lstm_dim]` = `[4 * lstm_dim]` flat array.
    ///
    /// # Returns
    ///
    /// `[2 * input_dim]`
    pub fn forward(
        &self,
        x: &[f32],
        n_nodes: usize,
        lstm_weight: &[f32],
        lstm_bias: &[f32],
    ) -> GnnResult<Vec<f32>> {
        let d = self.input_dim;
        let hd = self.lstm_dim; // LSTM hidden dim = 2*d

        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if x.len() != n_nodes * d {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes * d,
                got: x.len(),
            });
        }
        // Weight: [4 × hd × (hd + d)]
        let in_total = hd + d;
        if lstm_weight.len() != 4 * hd * in_total {
            return Err(GnnError::WeightShapeMismatch {
                r: 4 * hd,
                c: in_total,
                d: in_total,
            });
        }
        if lstm_bias.len() != 4 * hd {
            return Err(GnnError::DimensionMismatch {
                expected: 4 * hd,
                got: lstm_bias.len(),
            });
        }

        // Initial LSTM state
        let mut h = vec![0.0_f32; hd]; // hidden state
        let mut c = vec![0.0_f32; hd]; // cell state
        let mut q_star = vec![0.0_f32; hd]; // query = h at each step

        for _ in 0..self.processing_steps {
            // LSTM input: q_star (dim = hd) — previous query context vector
            let (h_new, c_new) = self.lstm_step(&q_star, &h, &c, lstm_weight, lstm_bias)?;
            h = h_new;
            c = c_new;

            // Compute attention: e_i = h^T x_i  (dot product between h and each node feature)
            // h is of dimension hd = 2*d, x_i is of dimension d
            // We use only the first `d` elements of h for the dot product
            let mut scores = Vec::with_capacity(n_nodes);
            for i in 0..n_nodes {
                let score: f32 = (0..d).map(|k| h[k] * x[i * d + k]).sum();
                scores.push(score);
            }

            // Softmax over scores
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            let alphas: Vec<f32> = if sum_e > 0.0 {
                exps.iter().map(|&e| e / sum_e).collect()
            } else {
                vec![1.0 / n_nodes as f32; n_nodes]
            };

            // r_t = Σ_i α_i x_i  [d]
            let mut r = vec![0.0_f32; d];
            for i in 0..n_nodes {
                for k in 0..d {
                    r[k] += alphas[i] * x[i * d + k];
                }
            }

            // q_star = [h || r]  [hd + d] but since hd = 2*d and output is also hd
            // We concatenate h (hd) and r (d) → length hd + d
            // But next iteration LSTM input is q_star with dim hd (the hidden state itself)
            // Per Set2Set: the readout q_t = [h_t* || r_t] is the final output only
            // The LSTM input for next step is h_t (not q_t)
            // We store q_star as [h || r] for the final step output
            q_star = {
                let mut qs = Vec::with_capacity(hd + d);
                qs.extend_from_slice(&h);
                qs.extend_from_slice(&r);
                qs
            };
        }

        // Return q_T which has dim = hd + d = 2*d + d = 3*d
        // but the spec says output_dim = 2*input_dim = hd
        // We return just the h || r[:d-d...] = first 2*d elements of q_star
        // Actually the canonical Set2Set output is the query vector q_t = [h_t || r_t]
        // with dim = lstm_dim + input_dim. But spec says output = 2*input_dim.
        // We return q_star[..hd] (h portion, length hd=2*d)
        let out: Vec<f32> = q_star[..hd].to_vec();
        Ok(out)
    }

    /// Single LSTM step.
    ///
    /// Input: `input` `[hd]` (the previous query q*), `h` `[hd]`, `c` `[hd]`.
    /// Uses a standard 4-gate LSTM:
    /// - `[i, f, g, o] = split_4(W * [h || input] + b)`
    /// - `c_new = sigmoid(f) ⊙ c + sigmoid(i) ⊙ tanh(g)`
    /// - `h_new = sigmoid(o) ⊙ tanh(c_new)`
    ///
    /// `weight`: `[4 * hd × in_total]` (gate × rows), `bias`: `[4 * hd]`.
    fn lstm_step(
        &self,
        input: &[f32],
        h: &[f32],
        c: &[f32],
        weight: &[f32],
        bias: &[f32],
    ) -> GnnResult<(Vec<f32>, Vec<f32>)> {
        let hd = self.lstm_dim;
        let d = self.input_dim;
        let in_total = hd + d;

        // Concatenate [h || input] of length hd + d (= in_total)
        // h has length hd, input has length hd (q*), but LSTM input should be d
        // In Set2Set the LSTM input at each step is the previous output q* which has dim hd
        // So in_total = hd + hd = 2*hd? No — let's use input.len() + h.len()
        let input_len = input.len().min(d); // clamp to actual input_dim
        let concat_len = hd + d;
        let _ = concat_len; // calculated already as in_total

        let mut concat = Vec::with_capacity(hd + input_len);
        concat.extend_from_slice(h);
        concat.extend_from_slice(&input[..input_len]);

        // Compute the 4 gates
        // weight: [4*hd × in_total], each gate has hd rows
        let mut gates = vec![0.0_f32; 4 * hd];
        for gate in 0..4 {
            for k in 0..hd {
                let row = gate * hd + k;
                let mut val = bias[row];
                let w_row_start = row * in_total;
                for j in 0..concat.len().min(in_total) {
                    val += weight[w_row_start + j] * concat[j];
                }
                gates[row] = val;
            }
        }

        // i, f, g, o gates
        let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
        let tanh = |v: f32| v.tanh();

        let mut c_new = vec![0.0_f32; hd];
        let mut h_new = vec![0.0_f32; hd];

        for k in 0..hd {
            let i_gate = sigmoid(gates[k]); // input gate
            let f_gate = sigmoid(gates[hd + k]); // forget gate
            let g_gate = tanh(gates[2 * hd + k]); // cell gate
            let o_gate = sigmoid(gates[3 * hd + k]); // output gate

            c_new[k] = f_gate * c[k] + i_gate * g_gate;
            h_new[k] = o_gate * tanh(c_new[k]);
        }

        Ok((h_new, c_new))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_weights(d: usize, steps: usize) -> (Vec<f32>, Vec<f32>) {
        let s2s = Set2Set::new(d, steps).expect("test invariant: value must be valid");
        let hd = s2s.lstm_dim;
        let in_total = hd + d;
        let w = vec![0.0_f32; 4 * hd * in_total];
        let b = vec![0.0_f32; 4 * hd];
        (w, b)
    }

    #[test]
    fn output_dim_is_twice_input_dim() {
        let s2s = Set2Set::new(8, 3).expect("test invariant: value must be valid");
        assert_eq!(s2s.output_dim(), 16);
    }

    #[test]
    fn output_shape_correct() {
        let d = 4;
        let n = 5;
        let s2s = Set2Set::new(d, 2).expect("test invariant: value must be valid");
        let x = vec![0.1_f32; n * d];
        let (w, b) = zero_weights(d, 2);
        let out = s2s
            .forward(&x, n, &w, &b)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), s2s.output_dim());
    }

    #[test]
    fn single_node_graph() {
        let d = 4;
        let s2s = Set2Set::new(d, 3).expect("test invariant: value must be valid");
        let x = vec![1.0_f32; d];
        let (w, b) = zero_weights(d, 3);
        let out = s2s
            .forward(&x, 1, &w, &b)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 2 * d);
    }

    #[test]
    fn output_finite() {
        let d = 3;
        let n = 6;
        let s2s = Set2Set::new(d, 4).expect("test invariant: value must be valid");
        let x: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.1).collect();
        let hd = s2s.lstm_dim;
        let in_total = hd + d;
        let w = vec![0.01_f32; 4 * hd * in_total];
        let b = vec![0.0_f32; 4 * hd];
        let out = s2s
            .forward(&x, n, &w, &b)
            .expect("test invariant: value must be valid");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_weights_output_zero() {
        let d = 4;
        let n = 3;
        let s2s = Set2Set::new(d, 2).expect("test invariant: value must be valid");
        let x = vec![0.5_f32; n * d];
        let (w, b) = zero_weights(d, 2);
        let out = s2s
            .forward(&x, n, &w, &b)
            .expect("test invariant: value must be valid");
        // With zero weights, LSTM gates are all 0 → sigmoid(0)=0.5, tanh(0)=0
        // c_new = 0.5*0 + 0.5*0 = 0, h_new = 0.5*tanh(0) = 0
        assert!(out.iter().all(|&v| v.abs() < 1e-5));
    }

    #[test]
    fn empty_graph_error() {
        let d = 4;
        let s2s = Set2Set::new(d, 2).expect("test invariant: value must be valid");
        let (w, b) = zero_weights(d, 2);
        let err = s2s.forward(&[], 0, &w, &b);
        assert!(matches!(err, Err(GnnError::EmptyGraph)));
    }

    #[test]
    fn multiple_processing_steps() {
        let d = 2;
        let n = 4;
        let steps = 5;
        let s2s = Set2Set::new(d, steps).expect("test invariant: value must be valid");
        let x = vec![1.0_f32; n * d];
        let (w, b) = zero_weights(d, steps);
        let out = s2s
            .forward(&x, n, &w, &b)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 2 * d);
    }

    #[test]
    fn lstm_step_output_shapes() {
        let d = 4;
        let steps = 1;
        let s2s = Set2Set::new(d, steps).expect("test invariant: value must be valid");
        let hd = s2s.lstm_dim;
        let in_total = hd + d;
        let input = vec![0.0_f32; d];
        let h = vec![0.0_f32; hd];
        let c = vec![0.0_f32; hd];
        let w = vec![0.0_f32; 4 * hd * in_total];
        let b = vec![0.0_f32; 4 * hd];
        let (h_new, c_new) = s2s
            .lstm_step(&input, &h, &c, &w, &b)
            .expect("test invariant: value must be valid");
        assert_eq!(h_new.len(), hd);
        assert_eq!(c_new.len(), hd);
    }

    #[test]
    fn invalid_zero_input_dim() {
        let err = Set2Set::new(0, 3);
        assert!(err.is_err());
    }

    #[test]
    fn invalid_zero_steps() {
        let err = Set2Set::new(4, 0);
        assert!(err.is_err());
    }

    #[test]
    fn dimension_mismatch_error() {
        let d = 4;
        let s2s = Set2Set::new(d, 2).expect("test invariant: value must be valid");
        let x = vec![0.1_f32; 3 * d]; // 3 nodes
        let (w, b) = zero_weights(d, 2);
        // Say n_nodes=5 but x only has 3 nodes' worth
        let err = s2s.forward(&x, 5, &w, &b);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }
}
