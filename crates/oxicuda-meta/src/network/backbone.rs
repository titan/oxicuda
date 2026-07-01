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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MetaError;
    use crate::handle::LcgRng;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    // -----------------------------------------------------------------------
    // Output-shape tests
    // -----------------------------------------------------------------------

    #[test]
    fn output_dim_matches_configured_last_dim() {
        let mut rng = make_rng(42);
        let dims = [4_usize, 8, 3];
        let net = MlpBackbone::new(&dims, &mut rng).expect("new ok");
        let x = vec![0.5_f32; 4];
        let out = net.forward(&x).expect("forward ok");
        assert_eq!(out.len(), 3, "output length must equal last dims entry");
    }

    // -----------------------------------------------------------------------
    // Determinism tests
    // -----------------------------------------------------------------------

    #[test]
    fn forward_is_deterministic_for_same_input() {
        let mut rng = make_rng(7);
        let net = MlpBackbone::new(&[4, 8, 3], &mut rng).expect("new ok");
        let x: Vec<f32> = (0..4).map(|i| i as f32 * 0.1).collect();
        let a = net.forward(&x).expect("first forward");
        let b = net.forward(&x).expect("second forward");
        assert_eq!(
            a, b,
            "same input must yield identical embedding on repeated call"
        );
    }

    // -----------------------------------------------------------------------
    // Numerical-health tests
    // -----------------------------------------------------------------------

    #[test]
    fn forward_outputs_are_all_finite() {
        let mut rng = make_rng(99);
        let net = MlpBackbone::new(&[16, 32, 8], &mut rng).expect("new ok");
        let x: Vec<f32> = (0..16).map(|i| (i as f32).sin()).collect();
        let out = net.forward(&x).expect("forward ok");
        for &v in &out {
            assert!(v.is_finite(), "all outputs must be finite, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // ReLU-placement tests
    // -----------------------------------------------------------------------

    #[test]
    fn relu_clamps_negative_preactivation_in_hidden_layer() {
        // Net [1, 1, 1].  Layer-0: w=-1, b=0 → pre-activation = -1*1 = -1.
        // ReLU must clamp the hidden value to 0.  Layer-1: w=1, b=0 → output = 0.
        let mut rng = make_rng(0);
        let mut net = MlpBackbone::new(&[1, 1, 1], &mut rng).expect("new ok");
        // params layout per-layer: [w_flat …, b_flat …] then next layer
        net.from_params(&[-1.0_f32, 0.0, 1.0, 0.0])
            .expect("load params ok");
        let out = net.forward(&[1.0_f32]).expect("forward ok");
        assert_eq!(
            out[0], 0.0_f32,
            "ReLU must clamp -1 hidden pre-activation to 0; final output should be 0"
        );
    }

    #[test]
    fn relu_is_not_applied_on_output_layer() {
        // Net [1, 1, 1].  Layer-0: w=+1 → hidden=+1 (passes ReLU).
        // Layer-1: w=-1, b=0 → output = -1.  No ReLU on the final layer.
        let mut rng = make_rng(0);
        let mut net = MlpBackbone::new(&[1, 1, 1], &mut rng).expect("new ok");
        net.from_params(&[1.0_f32, 0.0, -1.0, 0.0])
            .expect("load params ok");
        let out = net.forward(&[1.0_f32]).expect("forward ok");
        assert!(
            out[0] < 0.0,
            "output layer must NOT apply ReLU; expected negative value, got {}",
            out[0]
        );
    }

    // -----------------------------------------------------------------------
    // Parameter-management tests
    // -----------------------------------------------------------------------

    #[test]
    fn param_count_matches_analytical_formula() {
        let mut rng = make_rng(13);
        // Layer 0: 4*8 = 32 weights + 8 biases = 40
        // Layer 1: 8*3 = 24 weights + 3 biases = 27
        // Total: 67
        let dims = [4_usize, 8, 3];
        let net = MlpBackbone::new(&dims, &mut rng).expect("new ok");
        let expected = (4 * 8 + 8) + (8 * 3 + 3);
        assert_eq!(
            net.param_count(),
            expected,
            "param_count does not match layer arithmetic"
        );
    }

    #[test]
    fn to_params_from_params_roundtrip_preserves_forward() {
        let mut rng = make_rng(17);
        let mut net = MlpBackbone::new(&[4, 6, 2], &mut rng).expect("new ok");
        let x = vec![0.1_f32, 0.2, 0.3, 0.4];
        let before = net.forward(&x).expect("forward before roundtrip");
        let params = net.to_params();
        // Corrupt, then restore original params.
        net.from_params(&vec![0.0_f32; params.len()])
            .expect("zero all params");
        net.from_params(&params).expect("restore params");
        let after = net.forward(&x).expect("forward after roundtrip");
        assert_eq!(
            before, after,
            "param roundtrip must restore identical forward outputs"
        );
    }

    // -----------------------------------------------------------------------
    // Error-handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn new_errors_on_fewer_than_two_dims() {
        let mut rng = make_rng(1);
        assert!(
            matches!(
                MlpBackbone::new(&[4], &mut rng),
                Err(MetaError::BackboneError { .. })
            ),
            "single-element dims must return BackboneError"
        );
    }

    #[test]
    fn forward_errors_on_wrong_input_length() {
        let mut rng = make_rng(2);
        let net = MlpBackbone::new(&[4, 8, 3], &mut rng).expect("new ok");
        assert!(
            matches!(
                net.forward(&[1.0_f32, 2.0]),
                Err(MetaError::DimensionMismatch { .. })
            ),
            "mismatched input length must return DimensionMismatch"
        );
    }
}
