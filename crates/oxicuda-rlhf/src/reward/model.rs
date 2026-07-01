use crate::error::{RlhfError, RlhfResult};
use crate::handle::LcgRng;

#[derive(Debug)]
pub struct RewardModel {
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    dims: Vec<usize>,
}

impl RewardModel {
    pub fn new(dims: &[usize], rng: &mut LcgRng) -> RlhfResult<Self> {
        if dims.len() < 2 {
            return Err(RlhfError::Internal {
                msg: "dims must have at least 2 entries (input and output)".into(),
            });
        }
        let layers = dims
            .windows(2)
            .map(|pair| {
                let (in_d, out_d) = (pair[0], pair[1]);
                let scale = (2.0_f32 / in_d as f32).sqrt();
                let weights = (0..in_d * out_d)
                    .map(|_| {
                        let (a, _) = rng.next_normal_pair();
                        a * scale
                    })
                    .collect::<Vec<f32>>();
                let bias = vec![0.0_f32; out_d];
                (weights, bias)
            })
            .collect();
        Ok(Self {
            layers,
            dims: dims.to_vec(),
        })
    }

    pub fn forward(&self, x: &[f32]) -> RlhfResult<f32> {
        if x.len() != self.dims[0] {
            return Err(RlhfError::DimensionMismatch {
                expected: self.dims[0],
                got: x.len(),
            });
        }
        let mut current = x.to_vec();
        for (layer_idx, (weights, bias)) in self.layers.iter().enumerate() {
            let in_d = self.dims[layer_idx];
            let out_d = self.dims[layer_idx + 1];
            let mut next = vec![0.0_f32; out_d];
            for (j, (b, n)) in bias.iter().zip(next.iter_mut()).enumerate() {
                let row_start = j * in_d;
                let dot: f32 = weights[row_start..row_start + in_d]
                    .iter()
                    .zip(current.iter())
                    .map(|(&w, &c)| w * c)
                    .sum();
                *n = dot + b;
                if layer_idx + 1 < self.layers.len() {
                    *n = n.max(0.0);
                }
            }
            current = next;
        }
        let out = current[0];
        if out.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RlhfError;
    use crate::handle::LcgRng;

    // ── dims < 2 → Internal error ─────────────────────────────────────────────

    #[test]
    fn too_few_dims_errors() {
        let mut rng = LcgRng::new(1);
        let err = RewardModel::new(&[5], &mut rng).expect_err("fewer than 2 dims must error");
        assert!(
            matches!(err, RlhfError::Internal { .. }),
            "expected Internal error for single-dim spec, got {err:?}"
        );
    }

    // ── Wrong input length → DimensionMismatch ────────────────────────────────

    #[test]
    fn wrong_input_length_errors() {
        let mut rng = LcgRng::new(2);
        let model = RewardModel::new(&[3, 4, 1], &mut rng).expect("valid dims");
        let err = model
            .forward(&[1.0_f32, 2.0])
            .expect_err("input length 2 ≠ dim[0]=3 must error");
        assert!(
            matches!(
                err,
                RlhfError::DimensionMismatch {
                    expected: 3,
                    got: 2
                }
            ),
            "expected DimensionMismatch(3,2), got {err:?}"
        );
    }

    // ── Determinism: same seed + same input → identical scalar output ─────────

    #[test]
    fn forward_is_deterministic() {
        let mut rng_a = LcgRng::new(42);
        let model_a = RewardModel::new(&[4, 8, 1], &mut rng_a).expect("valid dims a");
        let mut rng_b = LcgRng::new(42);
        let model_b = RewardModel::new(&[4, 8, 1], &mut rng_b).expect("valid dims b");
        let input = [0.1_f32, 0.2, 0.3, 0.4];
        let out_a = model_a.forward(&input).expect("forward a");
        let out_b = model_b.forward(&input).expect("forward b");
        assert_eq!(
            out_a, out_b,
            "identical seed + input must yield identical output: {out_a} vs {out_b}"
        );
    }

    // ── Output is finite ──────────────────────────────────────────────────────

    #[test]
    fn forward_output_is_finite() {
        let mut rng = LcgRng::new(7);
        let model = RewardModel::new(&[3, 6, 2, 1], &mut rng).expect("valid dims");
        let mut inp_rng = LcgRng::new(99);
        let input: Vec<f32> = (0..3).map(|_| inp_rng.next_f32() * 2.0 - 1.0).collect();
        let out = model.forward(&input).expect("forward must succeed");
        assert!(out.is_finite(), "forward output must be finite, got {out}");
    }

    // ── Single-layer linear (W=1, b=0): forward(x) = x ───────────────────────

    #[test]
    fn single_layer_identity_weights() {
        // Manually construct a 1-input → 1-output model (no hidden layer).
        // The final (only) layer has no ReLU activation.
        // forward([x]) = 1.0 * x + 0.0 = x.
        let model = RewardModel {
            layers: vec![(vec![1.0_f32], vec![0.0_f32])],
            dims: vec![1, 1],
        };
        let out = model.forward(&[2.5_f32]).expect("identity forward");
        assert!(
            (out - 2.5).abs() < 1e-6,
            "identity W=1 b=0: expected 2.5, got {out}"
        );
        // Negative inputs also pass through unchanged — no ReLU on the output layer.
        let out_neg = model.forward(&[-3.0_f32]).expect("identity negative");
        assert!(
            (out_neg - (-3.0)).abs() < 1e-6,
            "identity W=1 b=0: expected -3.0 for negative input, got {out_neg}"
        );
    }

    // ── ReLU zeroes negative hidden activations; final layer is linear ─────────

    #[test]
    fn hidden_relu_zeroes_negatives() {
        // dims=[1, 2, 1]: one hidden layer (size 2) with ReLU, then linear output.
        //
        // Layer 0 (in_d=1, out_d=2); weights stored row-major (one row per output neuron):
        //   Neuron 0: weights[0..1] = [-1.0] → pre_relu = -1.0 * x
        //   Neuron 1: weights[1..2] = [ 1.0] → pre_relu =  1.0 * x
        //   Bias = [0.0, 0.0], ReLU applied (not the last layer).
        //
        // Layer 1 (in_d=2, out_d=1); weights stored row-major:
        //   Neuron 0: weights[0..2] = [0.0, 2.0] → dot = 0*h0 + 2*h1
        //   Bias = [0.0], no ReLU (last layer).
        //
        // x=1.0: pre_relu=[-1, 1] → relu=[0, 1] → output = 0*0 + 2*1 + 0 = 2.0
        // x=-1.0: pre_relu=[1, -1] → relu=[1, 0] → output = 0*1 + 2*0 + 0 = 0.0
        let model = RewardModel {
            layers: vec![
                (vec![-1.0_f32, 1.0], vec![0.0_f32, 0.0]), // layer 0: weights, bias
                (vec![0.0_f32, 2.0], vec![0.0_f32]),       // layer 1: weights, bias
            ],
            dims: vec![1, 2, 1],
        };
        let out_pos = model.forward(&[1.0_f32]).expect("relu hidden positive");
        assert!(
            (out_pos - 2.0).abs() < 1e-6,
            "ReLU hidden x=1: expected 2.0, got {out_pos}"
        );
        let out_neg = model.forward(&[-1.0_f32]).expect("relu hidden negative");
        assert!(
            (out_neg - 0.0).abs() < 1e-6,
            "ReLU hidden x=-1: negative pre-activation zeroed → expected 0.0, got {out_neg}"
        );
    }
}
