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

fn mlp_forward(x: &[f32], layers: &[(Vec<f32>, Vec<f32>)], input_dim: usize) -> f32 {
    let mut cur = x.to_vec();
    let mut cur_dim = input_dim;
    for (idx, (w, b)) in layers.iter().enumerate() {
        let out_dim = b.len();
        let mut out = dense(&cur, w, b, cur_dim, out_dim);
        if idx + 1 < layers.len() {
            for v in &mut out {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        cur = out;
        cur_dim = out_dim;
    }
    cur.first().copied().unwrap_or(0.0)
}

/// Entire Space Multi-Task Model (ESMM).
///
/// Models pCTR and pCVR jointly; pCTCVR = pCTR * pCVR to address selection bias.
pub struct Esmm {
    pub ctr_tower: Vec<(Vec<f32>, Vec<f32>)>,
    pub cvr_tower: Vec<(Vec<f32>, Vec<f32>)>,
    pub input_dim: usize,
}

impl Esmm {
    pub fn new(input_dim: usize, hidden_dims: &[usize], rng: &mut LcgRng) -> RecsysResult<Self> {
        if input_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: input_dim });
        }
        let build_tower = |rng: &mut LcgRng| -> Vec<(Vec<f32>, Vec<f32>)> {
            let mut layers = Vec::new();
            let mut in_dim = input_dim;
            for &out_dim in hidden_dims {
                let sc = (2.0 / in_dim as f32).sqrt();
                let w: Vec<f32> = (0..out_dim * in_dim)
                    .map(|_| rng.next_normal() * sc)
                    .collect();
                layers.push((w, vec![0.0_f32; out_dim]));
                in_dim = out_dim;
            }
            // Final scalar
            let sc = (2.0 / in_dim as f32).sqrt();
            let w: Vec<f32> = (0..in_dim).map(|_| rng.next_normal() * sc).collect();
            layers.push((w, vec![0.0_f32; 1]));
            layers
        };

        let ctr_tower = build_tower(rng);
        let cvr_tower = build_tower(rng);

        Ok(Self {
            ctr_tower,
            cvr_tower,
            input_dim,
        })
    }

    /// Returns (pCTR, pCVR, pCTCVR).
    pub fn forward(&self, x: &[f32]) -> RecsysResult<(f32, f32, f32)> {
        if x.len() != self.input_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let p_ctr = sigmoid(mlp_forward(x, &self.ctr_tower, self.input_dim));
        let p_cvr = sigmoid(mlp_forward(x, &self.cvr_tower, self.input_dim));
        let p_ctcvr = p_ctr * p_cvr;
        Ok((p_ctr, p_cvr, p_ctcvr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_esmm(seed: u64) -> Esmm {
        let mut rng = LcgRng::new(seed);
        Esmm::new(4, &[8, 4], &mut rng).expect("new ok")
    }

    fn sample_input(seed: u64, n: usize) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n).map(|_| rng.next_normal() * 0.5).collect()
    }

    #[test]
    fn pctcvr_equals_pctr_times_pcvr() {
        // ESMM's core identity (Ma et al. 2018): pCTCVR = pCTR × pCVR, by construction.
        // This test pins the implementation to that contract.
        let model = make_esmm(1);
        let x = sample_input(10, model.input_dim);
        let (p_ctr, p_cvr, p_ctcvr) = model.forward(&x).expect("fwd ok");
        let expected = p_ctr * p_cvr;
        assert!(
            (p_ctcvr - expected).abs() < 1e-7,
            "pCTCVR={p_ctcvr} must equal pCTR*pCVR={expected}"
        );
    }

    #[test]
    fn all_probabilities_in_unit_interval() {
        // sigmoid maps any logit into (0,1); the product of two (0,1) values is also
        // in [0,1].  All three outputs must satisfy this bound.
        let model = make_esmm(2);
        let x = sample_input(11, model.input_dim);
        let (p_ctr, p_cvr, p_ctcvr) = model.forward(&x).expect("fwd ok");
        assert!((0.0..=1.0).contains(&p_ctr), "pCTR={p_ctr} not in [0,1]");
        assert!((0.0..=1.0).contains(&p_cvr), "pCVR={p_cvr} not in [0,1]");
        assert!(
            (0.0..=1.0).contains(&p_ctcvr),
            "pCTCVR={p_ctcvr} not in [0,1]"
        );
    }

    #[test]
    fn pctcvr_leq_pctr() {
        // Because pCVR ∈ [0,1], pCTCVR = pCTR × pCVR ≤ pCTR.
        let model = make_esmm(3);
        let x = sample_input(12, model.input_dim);
        let (p_ctr, _p_cvr, p_ctcvr) = model.forward(&x).expect("fwd ok");
        assert!(
            p_ctcvr <= p_ctr + 1e-7,
            "pCTCVR={p_ctcvr} must be ≤ pCTR={p_ctr}"
        );
    }

    #[test]
    fn pctcvr_leq_pcvr() {
        // Because pCTR ∈ [0,1], pCTCVR = pCTR × pCVR ≤ pCVR.
        let model = make_esmm(4);
        let x = sample_input(13, model.input_dim);
        let (_p_ctr, p_cvr, p_ctcvr) = model.forward(&x).expect("fwd ok");
        assert!(
            p_ctcvr <= p_cvr + 1e-7,
            "pCTCVR={p_ctcvr} must be ≤ pCVR={p_cvr}"
        );
    }

    #[test]
    fn outputs_are_finite() {
        let model = make_esmm(5);
        let x = sample_input(14, model.input_dim);
        let (p_ctr, p_cvr, p_ctcvr) = model.forward(&x).expect("fwd ok");
        assert!(p_ctr.is_finite(), "pCTR is not finite");
        assert!(p_cvr.is_finite(), "pCVR is not finite");
        assert!(p_ctcvr.is_finite(), "pCTCVR is not finite");
    }

    #[test]
    fn determinism_same_seed_same_output() {
        let x = sample_input(99, 4);
        let out1 = make_esmm(6).forward(&x).expect("fwd");
        let out2 = make_esmm(6).forward(&x).expect("fwd");
        assert_eq!(out1.0.to_bits(), out2.0.to_bits(), "pCTR must be bit-exact");
        assert_eq!(out1.1.to_bits(), out2.1.to_bits(), "pCVR must be bit-exact");
        assert_eq!(
            out1.2.to_bits(),
            out2.2.to_bits(),
            "pCTCVR must be bit-exact"
        );
    }

    #[test]
    fn wrong_input_dim_returns_err() {
        let model = make_esmm(7);
        let bad = vec![0.0_f32; model.input_dim + 2];
        assert!(model.forward(&bad).is_err());
        assert!(model.forward(&[]).is_err());
    }

    #[test]
    fn zero_input_gives_half_for_ctr_and_cvr() {
        // Analytical invariant: all biases are 0, so MLP(0) = 0, and sigmoid(0) = 0.5.
        // Therefore pCTR = 0.5, pCVR = 0.5, and pCTCVR = 0.25 exactly.
        let model = make_esmm(8);
        let zeros = vec![0.0_f32; model.input_dim];
        let (p_ctr, p_cvr, p_ctcvr) = model.forward(&zeros).expect("fwd ok");
        assert!(
            (p_ctr - 0.5).abs() < 1e-6,
            "pCTR={p_ctr}, expected 0.5 on zero input"
        );
        assert!(
            (p_cvr - 0.5).abs() < 1e-6,
            "pCVR={p_cvr}, expected 0.5 on zero input"
        );
        assert!(
            (p_ctcvr - 0.25).abs() < 1e-6,
            "pCTCVR={p_ctcvr}, expected 0.25 on zero input"
        );
    }
}
