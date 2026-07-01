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

fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>() + 1e-10;
    exps.iter().map(|&e| e / sum).collect()
}

pub struct Mmoe {
    pub n_experts: usize,
    pub expert_dim: usize,
    pub input_dim: usize,
    /// Each expert: (W \[expert_dim x input_dim\], b \[expert_dim\])
    pub expert_layers: Vec<(Vec<f32>, Vec<f32>)>,
    /// Per-task gate weights: [n_experts x input_dim] per task
    pub gate_w: Vec<Vec<f32>>,
    /// Per-task towers: Vec of layers (W, b) -> scalar
    pub tower_layers: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
}

impl Mmoe {
    pub fn new(
        n_tasks: usize,
        n_experts: usize,
        expert_dim: usize,
        input_dim: usize,
        tower_hidden: usize,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if input_dim == 0 || expert_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: input_dim });
        }
        if n_experts == 0 {
            return Err(RecsysError::Internal {
                msg: "n_experts must be > 0".into(),
            });
        }
        let e_sc = (2.0 / input_dim as f32).sqrt();
        let expert_layers: Vec<(Vec<f32>, Vec<f32>)> = (0..n_experts)
            .map(|_| {
                let w: Vec<f32> = (0..expert_dim * input_dim)
                    .map(|_| rng.next_normal() * e_sc)
                    .collect();
                let b = vec![0.0_f32; expert_dim];
                (w, b)
            })
            .collect();

        let g_sc = (1.0 / input_dim as f32).sqrt();
        let gate_w: Vec<Vec<f32>> = (0..n_tasks)
            .map(|_| {
                (0..n_experts * input_dim)
                    .map(|_| rng.next_normal() * g_sc)
                    .collect()
            })
            .collect();

        let t_sc1 = (2.0 / expert_dim as f32).sqrt();
        let t_sc2 = (2.0 / tower_hidden as f32).sqrt();
        let tower_layers: Vec<Vec<(Vec<f32>, Vec<f32>)>> = (0..n_tasks)
            .map(|_| {
                vec![
                    {
                        let w: Vec<f32> = (0..tower_hidden * expert_dim)
                            .map(|_| rng.next_normal() * t_sc1)
                            .collect();
                        (w, vec![0.0_f32; tower_hidden])
                    },
                    {
                        let w: Vec<f32> = (0..tower_hidden)
                            .map(|_| rng.next_normal() * t_sc2)
                            .collect();
                        (w, vec![0.0_f32; 1])
                    },
                ]
            })
            .collect();

        Ok(Self {
            n_experts,
            expert_dim,
            input_dim,
            expert_layers,
            gate_w,
            tower_layers,
        })
    }

    pub fn forward(&self, x: &[f32]) -> RecsysResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }
        let d_e = self.expert_dim;

        // Expert outputs
        let expert_outs: Vec<Vec<f32>> = self
            .expert_layers
            .iter()
            .map(|(w, b)| {
                let mut out = dense(x, w, b, self.input_dim, d_e);
                for v in &mut out {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
                out
            })
            .collect();

        let n_tasks = self.gate_w.len();
        let mut task_outputs = Vec::with_capacity(n_tasks);

        for task in 0..n_tasks {
            // Gate: softmax(W_gate x) -> [n_experts]
            let gate_logits = dense(
                x,
                &self.gate_w[task],
                &vec![0.0_f32; self.n_experts],
                self.input_dim,
                self.n_experts,
            );
            let gate_weights = softmax(&gate_logits);

            // Weighted sum of expert outputs
            let mut mixed = vec![0.0_f32; d_e];
            for (e, (&gw, expert_out)) in gate_weights.iter().zip(expert_outs.iter()).enumerate() {
                let _ = e;
                for (m, &ev) in mixed.iter_mut().zip(expert_out.iter()) {
                    *m += gw * ev;
                }
            }

            // Tower MLP -> scalar
            let mut tower_cur = mixed;
            let mut cur_dim = d_e;
            for (idx, (w, b)) in self.tower_layers[task].iter().enumerate() {
                let out_dim = b.len();
                let mut out = dense(&tower_cur, w, b, cur_dim, out_dim);
                if idx + 1 < self.tower_layers[task].len() {
                    for v in &mut out {
                        if *v < 0.0 {
                            *v = 0.0;
                        }
                    }
                }
                tower_cur = out;
                cur_dim = out_dim;
            }

            let logit = tower_cur.first().copied().unwrap_or(0.0);
            task_outputs.push(sigmoid(logit));
        }

        Ok(task_outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_mmoe(seed: u64) -> Mmoe {
        let mut rng = LcgRng::new(seed);
        // 2 tasks, 3 experts, expert_dim=8, input_dim=4, tower_hidden=8
        Mmoe::new(2, 3, 8, 4, 8, &mut rng).expect("new ok")
    }

    fn sample_input(seed: u64, n: usize) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n).map(|_| rng.next_normal() * 0.5).collect()
    }

    #[test]
    fn output_len_matches_n_tasks() {
        let model = make_mmoe(1);
        let x = sample_input(10, model.input_dim);
        let out = model.forward(&x).expect("forward ok");
        assert_eq!(out.len(), model.gate_w.len());
    }

    #[test]
    fn all_outputs_in_open_unit_interval() {
        let model = make_mmoe(2);
        let x = sample_input(11, model.input_dim);
        let out = model.forward(&x).expect("forward ok");
        for (i, &v) in out.iter().enumerate() {
            assert!(v > 0.0 && v < 1.0, "task {i} output {v} not in (0,1)");
        }
    }

    #[test]
    fn outputs_are_finite() {
        let model = make_mmoe(3);
        let x = sample_input(12, model.input_dim);
        let out = model.forward(&x).expect("forward ok");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "task {i} output {v} is not finite");
        }
    }

    #[test]
    fn determinism_same_seed_same_output() {
        let x = sample_input(99, 4);
        let out1 = make_mmoe(5).forward(&x).expect("fwd");
        let out2 = make_mmoe(5).forward(&x).expect("fwd");
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "outputs must be bit-exact");
        }
    }

    #[test]
    fn wrong_input_dim_returns_err() {
        let model = make_mmoe(6);
        let bad = vec![0.0_f32; model.input_dim + 1];
        assert!(model.forward(&bad).is_err());
        assert!(model.forward(&[]).is_err());
    }

    #[test]
    fn invalid_config_returns_err() {
        let mut rng = LcgRng::new(7);
        assert!(
            Mmoe::new(2, 3, 8, 0, 8, &mut rng).is_err(),
            "zero input_dim"
        );
        assert!(
            Mmoe::new(2, 0, 8, 4, 8, &mut rng).is_err(),
            "zero n_experts"
        );
    }

    #[test]
    fn gate_weights_form_valid_probability_distribution() {
        // Using the public gate_w field, reconstruct each task's gate weights
        // as softmax(gate_w[t] @ x) and verify they are non-negative and sum to ~1.
        let model = make_mmoe(8);
        let x = sample_input(20, model.input_dim);
        let n_tasks = model.gate_w.len();
        let n_exp = model.n_experts;
        let in_d = model.input_dim;

        for t in 0..n_tasks {
            let logits: Vec<f32> = (0..n_exp)
                .map(|e| {
                    (0..in_d)
                        .map(|j| model.gate_w[t][e * in_d + j] * x[j])
                        .sum::<f32>()
                })
                .collect();
            // Numerically-stable softmax (same formula used in forward)
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum_e: f32 = exps.iter().sum::<f32>() + 1e-10;
            let weights: Vec<f32> = exps.iter().map(|&e| e / sum_e).collect();

            assert!(
                weights.iter().all(|&w| w >= 0.0),
                "task {t} gate has a negative weight"
            );
            let ws: f32 = weights.iter().sum();
            assert!(
                (ws - 1.0).abs() < 1e-5,
                "task {t} gate weights sum to {ws}, expected ~1.0"
            );
        }
    }

    #[test]
    fn identical_gate_and_tower_gives_equal_task_outputs() {
        // When two tasks share the same gate weights and the same tower, they must
        // produce identical outputs for any input (they compute the exact same function).
        let mut model = make_mmoe(9);
        model.gate_w[1] = model.gate_w[0].clone();
        model.tower_layers[1] = model.tower_layers[0].clone();
        let x = sample_input(21, model.input_dim);
        let out = model.forward(&x).expect("fwd ok");
        assert_eq!(out.len(), 2);
        assert!(
            (out[0] - out[1]).abs() < 1e-6,
            "identical gate+tower must produce identical outputs: {} vs {}",
            out[0],
            out[1]
        );
    }
}
