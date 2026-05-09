use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

pub struct MlpBackbone {
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
    dims: Vec<usize>,
}

impl MlpBackbone {
    pub fn new(dims: &[usize], rng: &mut LcgRng) -> MetaResult<Self> {
        if dims.len() < 2 {
            return Err(MetaError::BackboneError {
                msg: "dims must have at least 2 elements (input, output)".into(),
            });
        }
        for &d in dims {
            if d == 0 {
                return Err(MetaError::BackboneError {
                    msg: "all dims must be > 0".into(),
                });
            }
        }

        let n_layers = dims.len() - 1;
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let in_d = dims[l];
            let out_d = dims[l + 1];
            let limit = (6.0_f32 / (in_d + out_d) as f32).sqrt();
            let mut w = vec![0.0_f32; out_d * in_d];
            for v in w.iter_mut() {
                *v = (rng.next_f32() * 2.0 - 1.0) * limit;
            }
            weights.push(w);
            biases.push(vec![0.0_f32; out_d]);
        }

        Ok(Self {
            weights,
            biases,
            dims: dims.to_vec(),
        })
    }

    pub fn forward(&self, x: &[f32]) -> MetaResult<Vec<f32>> {
        let in_d = self.dims[0];
        if x.len() != in_d {
            return Err(MetaError::DimensionMismatch {
                expected: in_d,
                got: x.len(),
            });
        }

        let mut current = x.to_vec();
        let n_layers = self.dims.len() - 1;

        for l in 0..n_layers {
            let in_size = self.dims[l];
            let out_size = self.dims[l + 1];
            let w = &self.weights[l];
            let b = &self.biases[l];
            let mut next = vec![0.0_f32; out_size];

            for (o, (out_v, bias_v)) in next.iter_mut().zip(b.iter()).enumerate() {
                let row = &w[o * in_size..(o + 1) * in_size];
                *out_v = row
                    .iter()
                    .zip(current.iter())
                    .map(|(&wi, &xi)| wi * xi)
                    .sum::<f32>()
                    + bias_v;
            }

            // ReLU on all layers except the last
            if l < n_layers - 1 {
                for v in next.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }

            current = next;
        }

        Ok(current)
    }

    pub fn param_count(&self) -> usize {
        self.weights.iter().map(|w| w.len()).sum::<usize>()
            + self.biases.iter().map(|b| b.len()).sum::<usize>()
    }

    pub fn to_params(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.param_count());
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            out.extend_from_slice(w);
            out.extend_from_slice(b);
        }
        out
    }

    pub fn from_params(&mut self, params: &[f32]) -> MetaResult<()> {
        if params.len() != self.param_count() {
            return Err(MetaError::DimensionMismatch {
                expected: self.param_count(),
                got: params.len(),
            });
        }
        let mut offset = 0;
        for (w, b) in self.weights.iter_mut().zip(self.biases.iter_mut()) {
            let wlen = w.len();
            w.copy_from_slice(&params[offset..offset + wlen]);
            offset += wlen;
            let blen = b.len();
            b.copy_from_slice(&params[offset..offset + blen]);
            offset += blen;
        }
        Ok(())
    }
}
