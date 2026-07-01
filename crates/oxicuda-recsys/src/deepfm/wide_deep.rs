use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub struct WideDeep {
    pub wide_w: Vec<f32>,
    pub deep_layers: Vec<(Vec<f32>, Vec<f32>)>,
    pub input_dim: usize,
}

impl WideDeep {
    pub fn new(
        input_dim: usize,
        deep_hidden_dims: &[usize],
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if input_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: input_dim });
        }
        let wide_w: Vec<f32> = (0..input_dim).map(|_| rng.next_normal() * 0.01).collect();

        let mut deep_layers = Vec::new();
        let mut in_dim = input_dim;
        for &out_dim in deep_hidden_dims {
            let sc = (2.0 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..out_dim * in_dim)
                .map(|_| rng.next_normal() * sc)
                .collect();
            let b = vec![0.0_f32; out_dim];
            deep_layers.push((w, b));
            in_dim = out_dim;
        }
        // Final scalar output
        {
            let sc = (2.0 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..in_dim).map(|_| rng.next_normal() * sc).collect();
            let b = vec![0.0_f32; 1];
            deep_layers.push((w, b));
        }

        Ok(Self {
            wide_w,
            deep_layers,
            input_dim,
        })
    }

    pub fn forward(&self, x: &[f32]) -> RecsysResult<f32> {
        if x.len() != self.input_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        // Wide: linear dot product
        let wide_val: f32 = x
            .iter()
            .zip(self.wide_w.iter())
            .map(|(&xi, &wi)| xi * wi)
            .sum();

        // Deep: MLP with ReLU
        let mut deep_cur = x.to_vec();
        let mut cur_dim = self.input_dim;
        for (idx, (w, b)) in self.deep_layers.iter().enumerate() {
            let out_dim = b.len();
            let mut out = dense(&deep_cur, w, b, cur_dim, out_dim);
            if idx + 1 < self.deep_layers.len() {
                for v in &mut out {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            deep_cur = out;
            cur_dim = out_dim;
        }
        let deep_val = deep_cur.first().copied().unwrap_or(0.0);

        Ok(sigmoid(wide_val + deep_val))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    fn tiny_model(rng: &mut LcgRng) -> WideDeep {
        WideDeep::new(8, &[16, 8], rng).expect("must build")
    }

    #[test]
    fn output_in_unit_interval() {
        let mut rng = make_rng(1);
        let model = tiny_model(&mut rng);
        let x: Vec<f32> = (1..=8).map(|i| i as f32 * 0.1).collect();
        let p = model.forward(&x).expect("forward must succeed");
        assert!((0.0..=1.0).contains(&p), "output {p} not in [0,1]");
    }

    #[test]
    fn deterministic_same_seed() {
        let mut rng = make_rng(2);
        let model = tiny_model(&mut rng);
        let x = vec![0.5_f32; 8];
        let p1 = model.forward(&x).expect("must succeed");
        let p2 = model.forward(&x).expect("must succeed");
        assert_eq!(p1, p2, "same input must yield identical output");
    }

    #[test]
    fn finite_output() {
        let mut rng = make_rng(3);
        let model = tiny_model(&mut rng);
        let x: Vec<f32> = (0..8).map(|_| rng.next_f32()).collect();
        let p = model.forward(&x).expect("forward must succeed");
        assert!(p.is_finite(), "output must be finite, got {p}");
    }

    /// The wide part is a pure dot product, so it contributes linearly to the logit.
    /// With the deep weights zeroed (deep_val = 0), the logit equals wide_val = x·wide_w.
    /// Doubling x therefore doubles the logit: logit(2x) = 2·logit(x).
    /// Verify by recovering logits via inv_sigmoid(p) = ln(p/(1-p)).
    #[test]
    fn wide_linearity_doubling_input_doubles_logit() {
        let mut rng = make_rng(20);
        let mut model = WideDeep::new(4, &[8, 4], &mut rng).expect("must build");

        // Zero all deep weights so deep_val = 0 for any input.
        for (w, b) in &mut model.deep_layers {
            for v in w.iter_mut() {
                *v = 0.0;
            }
            for v in b.iter_mut() {
                *v = 0.0;
            }
        }
        // Set known wide weights.
        model.wide_w = vec![1.0, 0.5, 0.25, 0.125];

        let x1 = vec![0.2_f32, 0.4, 0.6, 0.8];
        let x2: Vec<f32> = x1.iter().map(|&v| 2.0 * v).collect();

        let p1 = model.forward(&x1).expect("forward must succeed");
        let p2 = model.forward(&x2).expect("forward must succeed");

        // wide_val_1 = 1·0.2 + 0.5·0.4 + 0.25·0.6 + 0.125·0.8 = 0.65
        // wide_val_2 = 2·0.65 = 1.30  → logit(2x) = 2·logit(x)
        let logit1 = (p1 / (1.0 - p1)).ln();
        let logit2 = (p2 / (1.0 - p2)).ln();
        assert!(
            (logit2 - 2.0 * logit1).abs() < 1e-4,
            "wide linearity: logit(2x)={logit2}, 2·logit(x)={:.6}",
            2.0 * logit1
        );
    }

    /// The Wide&Deep logit is wide_val + deep_val (before sigmoid). Therefore,
    /// the three logits of the full model, wide-only (deep zeroed), and deep-only
    /// (wide zeroed) must satisfy: logit_full = logit_wide + logit_deep.
    /// All three models are built from the same seed so their weights are identical.
    #[test]
    fn additive_combination_logit_equals_wide_plus_deep() {
        let seed = 30_u64;
        let x = vec![0.1_f32, 0.2, 0.3, 0.4];

        let mut rng_full = LcgRng::new(seed);
        let model_full = WideDeep::new(4, &[8], &mut rng_full).expect("must build");
        let p_full = model_full.forward(&x).expect("forward must succeed");

        // Same weights, deep zeroed → logit = wide_val
        let mut rng_wide = LcgRng::new(seed);
        let mut model_wide = WideDeep::new(4, &[8], &mut rng_wide).expect("must build");
        for (w, b) in &mut model_wide.deep_layers {
            for v in w.iter_mut() {
                *v = 0.0;
            }
            for v in b.iter_mut() {
                *v = 0.0;
            }
        }
        let p_wide = model_wide.forward(&x).expect("forward must succeed");

        // Same weights, wide zeroed → logit = deep_val
        let mut rng_deep = LcgRng::new(seed);
        let mut model_deep = WideDeep::new(4, &[8], &mut rng_deep).expect("must build");
        for v in &mut model_deep.wide_w {
            *v = 0.0;
        }
        let p_deep = model_deep.forward(&x).expect("forward must succeed");

        // Recover logits via inv_sigmoid = ln(p / (1-p)).
        let logit_full = (p_full / (1.0 - p_full)).ln();
        let logit_wide = (p_wide / (1.0 - p_wide)).ln();
        let logit_deep = (p_deep / (1.0 - p_deep)).ln();

        assert!(
            (logit_full - logit_wide - logit_deep).abs() < 1e-4,
            "additive: full={logit_full:.6}, wide={logit_wide:.6}, deep={logit_deep:.6}"
        );
    }

    #[test]
    fn wrong_input_length_errors() {
        let mut rng = make_rng(4);
        let model = tiny_model(&mut rng); // input_dim = 8
        let x = vec![0.0_f32; 5]; // 5 ≠ 8 → mismatch
        let err = model.forward(&x);
        assert!(matches!(err, Err(RecsysError::DimensionMismatch { .. })));
    }

    #[test]
    fn zero_input_dim_rejected() {
        let mut rng = make_rng(5);
        let err = WideDeep::new(0, &[8], &mut rng);
        assert!(matches!(err, Err(RecsysError::InvalidEmbeddingDim { .. })));
    }
}
