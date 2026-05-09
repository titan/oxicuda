use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn dense(x: &[f32], w: &[f32], b: &[f32], fan_in: usize, fan_out: usize) -> Vec<f32> {
    (0..fan_out)
        .map(|o| {
            b[o] + w[o * fan_in..(o + 1) * fan_in]
                .iter()
                .zip(x.iter())
                .map(|(&wi, &xi)| wi * xi)
                .sum::<f32>()
        })
        .collect()
}

fn init_layer(fan_in: usize, fan_out: usize, rng: &mut LcgRng) -> (Vec<f32>, Vec<f32>) {
    let scale = (2.0_f32 / fan_in as f32).sqrt();
    let w = (0..fan_in * fan_out)
        .map(|_| rng.next_normal() * scale)
        .collect();
    let b = vec![0.0_f32; fan_out];
    (w, b)
}

fn mlp_forward(
    layers: &[(Vec<f32>, Vec<f32>)],
    x: &[f32],
    dims: &[usize],
    activate_last: bool,
) -> Vec<f32> {
    let mut h = x.to_vec();
    let n_layers = layers.len();
    for (idx, (w, b)) in layers.iter().enumerate() {
        let fan_in = dims[idx];
        let fan_out = dims[idx + 1];
        let out = dense(&h, w, b, fan_in, fan_out);
        let is_last = idx == n_layers - 1;
        h = if is_last && !activate_last {
            out
        } else {
            out.iter().map(|&v| relu(v)).collect()
        };
    }
    h
}

fn mlp_backward_update(
    layers: &mut [(Vec<f32>, Vec<f32>)],
    x: &[f32],
    error: f32,
    dims: &[usize],
    lr: f32,
) {
    // Simple one-step gradient update for a single-output MLP via backprop
    let n_layers = layers.len();
    let mut activations: Vec<Vec<f32>> = Vec::with_capacity(n_layers + 1);
    activations.push(x.to_vec());

    let mut h = x.to_vec();
    for (idx, (w, b)) in layers.iter().enumerate() {
        let fan_in = dims[idx];
        let fan_out = dims[idx + 1];
        let pre = dense(&h, w, b, fan_in, fan_out);
        let is_last = idx == n_layers - 1;
        h = if is_last {
            pre.clone()
        } else {
            pre.iter().map(|&v| relu(v)).collect()
        };
        activations.push(h.clone());
    }

    // Backprop: output error -> each layer
    let mut delta: Vec<f32> = vec![error];
    for layer_idx in (0..n_layers).rev() {
        let fan_in = dims[layer_idx];
        let fan_out = dims[layer_idx + 1];
        let (w, b) = &mut layers[layer_idx];
        let h_prev = &activations[layer_idx];
        let h_curr = &activations[layer_idx + 1];

        let mut new_delta = vec![0.0_f32; fan_in];
        for o in 0..fan_out {
            let d_out = delta[o];
            // gradient through activation (relu gate)
            let gate = if layer_idx < n_layers - 1 {
                if h_curr[o] > 0.0 { 1.0 } else { 0.0 }
            } else {
                1.0
            };
            let d = d_out * gate;
            b[o] -= lr * d;
            for i in 0..fan_in {
                w[o * fan_in + i] -= lr * d * h_prev[i];
                new_delta[i] += d * w[o * fan_in + i];
            }
        }
        // Apply relu gate to new_delta based on previous layer activations
        if layer_idx > 0 {
            let h_gate = &activations[layer_idx];
            for (nd, &hg) in new_delta.iter_mut().zip(h_gate.iter()) {
                if hg <= 0.0 {
                    *nd = 0.0;
                }
            }
        }
        delta = new_delta;
    }
}

/// DeepIV: two-stage neural network instrumental variable estimator.
pub struct DeepIv {
    stage1_w: Vec<(Vec<f32>, Vec<f32>)>,
    stage1_dims: Vec<usize>,
    stage2_w: Vec<(Vec<f32>, Vec<f32>)>,
    stage2_dims: Vec<usize>,
    pub input_dim: usize,
    pub n_instruments: usize,
}

impl DeepIv {
    pub fn new(
        input_dim: usize,
        n_instruments: usize,
        hidden_dim: usize,
        n_layers: usize,
        rng: &mut LcgRng,
    ) -> Self {
        let mut stage1_w = Vec::with_capacity(n_layers + 1);
        let mut stage1_dims = vec![n_instruments];
        let mut prev = n_instruments;
        for _ in 0..n_layers {
            stage1_w.push(init_layer(prev, hidden_dim, rng));
            stage1_dims.push(hidden_dim);
            prev = hidden_dim;
        }
        stage1_w.push(init_layer(prev, 1, rng));
        stage1_dims.push(1);

        let mut stage2_w = Vec::with_capacity(n_layers + 1);
        let mut stage2_dims = vec![input_dim + 1]; // [x, t_hat]
        prev = input_dim + 1;
        for _ in 0..n_layers {
            stage2_w.push(init_layer(prev, hidden_dim, rng));
            stage2_dims.push(hidden_dim);
            prev = hidden_dim;
        }
        stage2_w.push(init_layer(prev, 1, rng));
        stage2_dims.push(1);

        Self {
            stage1_w,
            stage1_dims,
            stage2_w,
            stage2_dims,
            input_dim,
            n_instruments,
        }
    }

    /// Stage 1: predict T from Z.
    fn predict_t_from_z(&self, z: &[f32]) -> f32 {
        let out = mlp_forward(&self.stage1_w, z, &self.stage1_dims, false);
        out[0]
    }

    /// Stage 2: predict Y from [X, T_hat].
    fn predict_y(&self, x: &[f32], t_hat: f32) -> f32 {
        let mut inp = x.to_vec();
        inp.push(t_hat);
        let out = mlp_forward(&self.stage2_w, &inp, &self.stage2_dims, false);
        out[0]
    }

    pub fn fit_stage1(
        &mut self,
        z: &[f32],
        t: &[f32],
        n: usize,
        lr: f32,
        n_epochs: usize,
    ) -> CausalResult<()> {
        if z.is_empty() || n == 0 {
            return Err(CausalError::EmptyInput);
        }
        if z.len() != n * self.n_instruments || t.len() != n {
            return Err(CausalError::IncompatibleData);
        }
        for _ in 0..n_epochs {
            for i in 0..n {
                let zi = &z[i * self.n_instruments..(i + 1) * self.n_instruments];
                let t_pred = self.predict_t_from_z(zi);
                let err = t_pred - t[i];
                mlp_backward_update(&mut self.stage1_w, zi, err, &self.stage1_dims, lr);
            }
        }
        Ok(())
    }

    pub fn fit_stage2(
        &mut self,
        x: &[f32],
        t_hat: &[f32],
        y: &[f32],
        n: usize,
        lr: f32,
        n_epochs: usize,
    ) -> CausalResult<()> {
        if x.is_empty() || n == 0 {
            return Err(CausalError::EmptyInput);
        }
        if x.len() != n * self.input_dim || t_hat.len() != n || y.len() != n {
            return Err(CausalError::IncompatibleData);
        }
        for _ in 0..n_epochs {
            for i in 0..n {
                let xi = &x[i * self.input_dim..(i + 1) * self.input_dim];
                let y_pred = self.predict_y(xi, t_hat[i]);
                let err = y_pred - y[i];
                let mut inp = xi.to_vec();
                inp.push(t_hat[i]);
                mlp_backward_update(&mut self.stage2_w, &inp, err, &self.stage2_dims, lr);
            }
        }
        Ok(())
    }

    pub fn predict_outcome(&self, x: &[f32], t: &[f32], n: usize) -> CausalResult<Vec<f32>> {
        if x.is_empty() || n == 0 {
            return Err(CausalError::EmptyInput);
        }
        if x.len() != n * self.input_dim || t.len() != n {
            return Err(CausalError::IncompatibleData);
        }
        Ok((0..n)
            .map(|i| {
                let xi = &x[i * self.input_dim..(i + 1) * self.input_dim];
                self.predict_y(xi, t[i])
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepiv_predict_shape() {
        let mut rng = LcgRng::new(7);
        let mut model = DeepIv::new(3, 2, 8, 2, &mut rng);
        let n = 10;
        let z: Vec<f32> = (0..n * 2).map(|i| i as f32 / 20.0).collect();
        let t: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        model.fit_stage1(&z, &t, n, 0.001, 5).unwrap();
        let x: Vec<f32> = (0..n * 3).map(|i| i as f32 / 30.0).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        model.fit_stage2(&x, &t, &y, n, 0.001, 5).unwrap();
        let out = model.predict_outcome(&x, &t, n).unwrap();
        assert_eq!(out.len(), n);
    }
}
