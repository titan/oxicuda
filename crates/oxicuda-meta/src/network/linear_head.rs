use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

pub struct LinearHead {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub n_classes: usize,
    pub feat_dim: usize,
}

impl LinearHead {
    pub fn new(feat_dim: usize, n_classes: usize, rng: &mut LcgRng) -> Self {
        let limit = (6.0_f32 / (feat_dim + n_classes) as f32).sqrt();
        let mut weights = vec![0.0_f32; n_classes * feat_dim];
        for v in weights.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * limit;
        }
        Self {
            weights,
            biases: vec![0.0_f32; n_classes],
            n_classes,
            feat_dim,
        }
    }

    pub fn forward(&self, feat: &[f32]) -> MetaResult<Vec<f32>> {
        if feat.len() != self.feat_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.feat_dim,
                got: feat.len(),
            });
        }
        let logits: Vec<f32> = (0..self.n_classes)
            .map(|c| {
                let row = &self.weights[c * self.feat_dim..(c + 1) * self.feat_dim];
                row.iter()
                    .zip(feat.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f32>()
                    + self.biases[c]
            })
            .collect();
        Ok(logits)
    }

    pub fn param_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    pub fn to_params(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.param_count());
        out.extend_from_slice(&self.weights);
        out.extend_from_slice(&self.biases);
        out
    }

    pub fn from_params(&mut self, params: &[f32]) -> MetaResult<()> {
        if params.len() != self.param_count() {
            return Err(MetaError::DimensionMismatch {
                expected: self.param_count(),
                got: params.len(),
            });
        }
        let wlen = self.weights.len();
        self.weights.copy_from_slice(&params[..wlen]);
        self.biases.copy_from_slice(&params[wlen..]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── param_count and to_params / from_params ───────────────────────────────

    #[test]
    fn param_count_matches_formula() {
        // param_count == n_classes * feat_dim + n_classes
        let mut rng = LcgRng::new(42);
        let head = LinearHead::new(3, 4, &mut rng);
        assert_eq!(head.param_count(), 3 * 4 + 4, "expected 3*4+4=16");
    }

    #[test]
    fn to_params_length_matches_param_count() {
        let mut rng = LcgRng::new(17);
        let head = LinearHead::new(5, 3, &mut rng);
        assert_eq!(head.to_params().len(), head.param_count());
    }

    #[test]
    fn to_params_layout_is_weights_then_biases() {
        // to_params() = [weights..., biases...]
        let mut rng = LcgRng::new(99);
        let head = LinearHead::new(2, 2, &mut rng);
        let params = head.to_params();
        // First n_classes*feat_dim entries are weights
        assert_eq!(&params[..head.weights.len()], head.weights.as_slice());
        // Remaining n_classes entries are biases
        assert_eq!(&params[head.weights.len()..], head.biases.as_slice());
    }

    #[test]
    fn from_params_round_trip() {
        let mut rng = LcgRng::new(77);
        let mut head = LinearHead::new(3, 4, &mut rng);
        let original = head.to_params();
        // Zero everything, then restore
        for v in head.weights.iter_mut() {
            *v = 0.0;
        }
        for v in head.biases.iter_mut() {
            *v = 0.0;
        }
        head.from_params(&original)
            .expect("from_params should succeed");
        assert_eq!(
            head.to_params(),
            original,
            "to_params must match after round-trip"
        );
    }

    // ── forward ───────────────────────────────────────────────────────────────

    #[test]
    fn forward_output_length_matches_n_classes() {
        let mut rng = LcgRng::new(7);
        let head = LinearHead::new(4, 5, &mut rng);
        let feat = vec![0.1_f32; 4];
        let logits = head.forward(&feat).expect("forward should succeed");
        assert_eq!(logits.len(), 5, "output length must equal n_classes");
    }

    #[test]
    fn forward_manual_computation() {
        // feat_dim=2, n_classes=2; weights row-major:
        //   W[0] = [2.0, -1.0],  b[0] = 0.1
        //   W[1] = [0.5,  1.0],  b[1] = -0.2
        // feat = [1.0, 3.0]
        //   logit[0] = 2.0*1.0 + (-1.0)*3.0 + 0.1 = -0.9
        //   logit[1] = 0.5*1.0 +  1.0*3.0 + (-0.2) = 3.3
        let mut rng = LcgRng::new(42);
        let mut head = LinearHead::new(2, 2, &mut rng);
        head.weights = vec![2.0_f32, -1.0, 0.5, 1.0];
        head.biases = vec![0.1_f32, -0.2];
        let feat = [1.0_f32, 3.0];
        let logits = head.forward(&feat).expect("forward should succeed");
        assert_eq!(logits.len(), 2);
        assert!(
            (logits[0] - (-0.9_f32)).abs() < 1e-5,
            "expected logit[0]=-0.9, got {}",
            logits[0]
        );
        assert!(
            (logits[1] - 3.3_f32).abs() < 1e-5,
            "expected logit[1]=3.3, got {}",
            logits[1]
        );
    }

    #[test]
    fn forward_dim_mismatch_errors() {
        let mut rng = LcgRng::new(5);
        let head = LinearHead::new(4, 3, &mut rng);
        assert!(matches!(
            head.forward(&[0.0_f32; 3]),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── determinism and initialisation ───────────────────────────────────────

    #[test]
    fn deterministic_with_fixed_seed() {
        let feat = vec![0.3_f32; 6];
        let mut rng_a = LcgRng::new(2026);
        let head_a = LinearHead::new(6, 4, &mut rng_a);
        let mut rng_b = LcgRng::new(2026);
        let head_b = LinearHead::new(6, 4, &mut rng_b);
        assert_eq!(
            head_a.forward(&feat).expect("forward a should succeed"),
            head_b.forward(&feat).expect("forward b should succeed"),
            "same seed must produce identical outputs"
        );
    }

    #[test]
    fn biases_zero_at_init() {
        let mut rng = LcgRng::new(1);
        let head = LinearHead::new(8, 5, &mut rng);
        assert!(
            head.biases.iter().all(|&b| b == 0.0_f32),
            "biases must be zero-initialised"
        );
    }
}
