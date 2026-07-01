use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// GRU4Rec: session-based recommendation using Gated Recurrent Units.
///
/// Weight layout for w_ih (input-to-hidden): [3 * hidden_dim x emb_dim]
/// Rows 0..hidden_dim: z gate weights
/// Rows hidden_dim..2*hidden_dim: r gate weights
/// Rows 2*hidden_dim..3*hidden_dim: n gate weights
///
/// Same layout for w_hh (hidden-to-hidden): [3 * hidden_dim x hidden_dim]
/// b_h: [3 * hidden_dim]
pub struct Gru4Rec {
    pub item_emb: Vec<f32>,
    pub n_items: usize,
    pub emb_dim: usize,
    pub hidden_dim: usize,
    /// [3 * hidden_dim x emb_dim]
    pub w_ih: Vec<f32>,
    /// [3 * hidden_dim x hidden_dim]
    pub w_hh: Vec<f32>,
    /// [3 * hidden_dim]
    pub b_h: Vec<f32>,
    /// [n_items x hidden_dim] (item scoring weights)
    pub output_w: Vec<f32>,
}

impl Gru4Rec {
    pub fn new(
        n_items: usize,
        emb_dim: usize,
        hidden_dim: usize,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: n_items });
        }
        if emb_dim == 0 || hidden_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: emb_dim });
        }
        let emb_scale = (1.0 / emb_dim as f32).sqrt();
        let ih_scale = (2.0 / emb_dim as f32).sqrt();
        let hh_scale = (2.0 / hidden_dim as f32).sqrt();

        let item_emb: Vec<f32> = (0..n_items * emb_dim)
            .map(|_| rng.next_normal() * emb_scale)
            .collect();
        let w_ih: Vec<f32> = (0..3 * hidden_dim * emb_dim)
            .map(|_| rng.next_normal() * ih_scale)
            .collect();
        let w_hh: Vec<f32> = (0..3 * hidden_dim * hidden_dim)
            .map(|_| rng.next_normal() * hh_scale)
            .collect();
        let b_h = vec![0.0_f32; 3 * hidden_dim];
        let out_scale = (2.0 / hidden_dim as f32).sqrt();
        let output_w: Vec<f32> = (0..n_items * hidden_dim)
            .map(|_| rng.next_normal() * out_scale)
            .collect();

        Ok(Self {
            item_emb,
            n_items,
            emb_dim,
            hidden_dim,
            w_ih,
            w_hh,
            b_h,
            output_w,
        })
    }

    fn gru_cell(&self, x: &[f32], h: &[f32]) -> Vec<f32> {
        let d_h = self.hidden_dim;
        let d_x = self.emb_dim;

        // Compute gate pre-activations
        let z_pre: Vec<f32> = (0..d_h)
            .map(|i| {
                self.b_h[i]
                    + self.w_ih[i * d_x..(i + 1) * d_x]
                        .iter()
                        .zip(x.iter())
                        .map(|(&w, &xi)| w * xi)
                        .sum::<f32>()
                    + self.w_hh[i * d_h..(i + 1) * d_h]
                        .iter()
                        .zip(h.iter())
                        .map(|(&w, &hi)| w * hi)
                        .sum::<f32>()
            })
            .collect();

        let r_pre: Vec<f32> = (0..d_h)
            .map(|i| {
                let row = d_h + i;
                self.b_h[row]
                    + self.w_ih[row * d_x..(row + 1) * d_x]
                        .iter()
                        .zip(x.iter())
                        .map(|(&w, &xi)| w * xi)
                        .sum::<f32>()
                    + self.w_hh[row * d_h..(row + 1) * d_h]
                        .iter()
                        .zip(h.iter())
                        .map(|(&w, &hi)| w * hi)
                        .sum::<f32>()
            })
            .collect();

        let z: Vec<f32> = z_pre.iter().map(|&v| sigmoid(v)).collect();
        let r: Vec<f32> = r_pre.iter().map(|&v| sigmoid(v)).collect();

        // n gate: tanh(Wn x + r * (Un h + bn))
        let n_pre: Vec<f32> = (0..d_h)
            .map(|i| {
                let row = 2 * d_h + i;
                let ih_part: f32 = self.b_h[row]
                    + self.w_ih[row * d_x..(row + 1) * d_x]
                        .iter()
                        .zip(x.iter())
                        .map(|(&w, &xi)| w * xi)
                        .sum::<f32>();
                let hh_part: f32 = self.w_hh[row * d_h..(row + 1) * d_h]
                    .iter()
                    .zip(h.iter())
                    .map(|(&w, &hi)| w * hi)
                    .sum::<f32>();
                ih_part + r[i] * hh_part
            })
            .collect();

        let n: Vec<f32> = n_pre.iter().map(|&v| v.tanh()).collect();

        // h' = (1 - z) * h + z * n
        (0..d_h)
            .map(|i| (1.0 - z[i]) * h[i] + z[i] * n[i])
            .collect()
    }

    pub fn forward(&self, item_ids: &[usize]) -> RecsysResult<Vec<f32>> {
        if item_ids.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        for &id in item_ids {
            if id >= self.n_items {
                return Err(RecsysError::UnknownItem { id });
            }
        }

        let d = self.emb_dim;
        let d_h = self.hidden_dim;
        let mut h = vec![0.0_f32; d_h];

        for &id in item_ids {
            let x = &self.item_emb[id * d..(id + 1) * d];
            h = self.gru_cell(x, &h);
        }

        // Compute logits over all items: [n_items]
        let logits: Vec<f32> = (0..self.n_items)
            .map(|item| {
                self.output_w[item * d_h..(item + 1) * d_h]
                    .iter()
                    .zip(h.iter())
                    .map(|(&w, &hi)| w * hi)
                    .sum()
            })
            .collect();

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(seed: u64) -> Gru4Rec {
        let mut rng = LcgRng::new(seed);
        Gru4Rec::new(10, 8, 16, &mut rng).expect("construction ok")
    }

    #[test]
    fn rejects_invalid_construction() {
        let mut rng = LcgRng::new(1);
        assert!(
            Gru4Rec::new(0, 8, 16, &mut rng).is_err(),
            "n_items=0 must fail"
        );
        assert!(
            Gru4Rec::new(10, 0, 16, &mut rng).is_err(),
            "emb_dim=0 must fail"
        );
        assert!(
            Gru4Rec::new(10, 8, 0, &mut rng).is_err(),
            "hidden_dim=0 must fail"
        );
    }

    #[test]
    fn output_shape_and_finiteness() {
        let model = make_model(2);
        let logits = model.forward(&[0, 1, 2]).expect("forward ok");
        assert_eq!(
            logits.len(),
            10,
            "forward must return exactly n_items=10 logits"
        );
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}]={v} must be finite");
        }
    }

    #[test]
    fn forward_rejects_empty_and_oob() {
        let model = make_model(3);
        assert!(model.forward(&[]).is_err(), "empty input must fail");
        assert!(
            model.forward(&[10]).is_err(),
            "id=10 >= n_items=10 must fail"
        );
        // Boundary: id=9 is the last valid item.
        assert!(model.forward(&[9]).is_ok(), "id=9 must succeed");
    }

    #[test]
    fn determinism() {
        // Two models from the same seed must produce bit-identical outputs.
        let model_a = make_model(5);
        let model_b = make_model(5);
        let seq = [0usize, 1, 2, 3, 4];
        let out_a = model_a.forward(&seq).expect("fwd a");
        let out_b = model_b.forward(&seq).expect("fwd b");
        for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
            assert_eq!(
                a, b,
                "logit[{i}] must be bit-identical across same-seed models"
            );
        }
    }

    #[test]
    fn zero_embedding_gives_zero_logits() {
        // Analytic fixed-point proof: if x=0 and h_0=0 then the GRU cell gives h'=0.
        //
        // With b_h=0, x=0, h=0:
        //   z_pre[i] = 0  →  z[i] = sigmoid(0) = 0.5
        //   r_pre[i] = 0  →  r[i] = sigmoid(0) = 0.5
        //   n_pre[i] = 0 + 0.5 * 0 = 0  →  n[i] = tanh(0) = 0
        //   h'[i] = (1 - 0.5)*0 + 0.5*0 = 0
        //
        // Therefore logits = output_w @ h' = 0.
        let emb_dim = 4usize;
        let hidden_dim = 6usize;
        let mut rng = LcgRng::new(99);
        let mut model = Gru4Rec::new(5, emb_dim, hidden_dim, &mut rng).expect("construction ok");
        // Zero item 0's embedding so x=0 on the single GRU step.
        for i in 0..emb_dim {
            model.item_emb[i] = 0.0;
        }
        // Biases are already 0 by construction; zero them explicitly to document the
        // assumption the analytic proof relies on.
        for b in &mut model.b_h {
            *b = 0.0;
        }
        let logits = model.forward(&[0]).expect("fwd");
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.abs() < 1e-6,
                "logit[{i}]={l}: zero embedding + zero bias must give zero hidden state and zero logits"
            );
        }
    }

    #[test]
    fn gru_updates_state_across_steps() {
        // Processing the same item twice must yield a different hidden state than once:
        // the second step takes the first step's non-zero h as the recurrent input,
        // and the z gate blends it with the new candidate n, producing a different h'.
        let model = make_model(6);
        let out_once = model.forward(&[0]).expect("fwd [0]");
        let out_twice = model.forward(&[0, 0]).expect("fwd [0, 0]");
        let differs = out_once
            .iter()
            .zip(out_twice.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "processing the same item twice must update the hidden state (GRU recurrence)"
        );
    }

    #[test]
    fn order_sensitivity() {
        // GRU processes items strictly left-to-right; permuting the input changes
        // the intermediate hidden states and therefore the final logit vector.
        let model = make_model(7);
        let out_01 = model.forward(&[0, 1]).expect("fwd [0,1]");
        let out_10 = model.forward(&[1, 0]).expect("fwd [1,0]");
        let differs = out_01
            .iter()
            .zip(out_10.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "GRU must be order-sensitive: forward([0,1]) != forward([1,0])"
        );
    }
}
