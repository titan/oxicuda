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
