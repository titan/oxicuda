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
