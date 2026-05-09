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

fn relu_vec(mut v: Vec<f32>) -> Vec<f32> {
    for x in &mut v {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
    v
}

/// Expert network: single linear layer with ReLU.
struct Expert {
    w: Vec<f32>,
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Expert {
    fn new(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        let sc = (2.0 / in_dim as f32).sqrt();
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| rng.next_normal() * sc)
            .collect();
        Self {
            w,
            b: vec![0.0_f32; out_dim],
            in_dim,
            out_dim,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        relu_vec(dense(x, &self.w, &self.b, self.in_dim, self.out_dim))
    }
}

/// Gating network: softmax over expert candidates.
struct Gate {
    w: Vec<f32>,
    in_dim: usize,
    n_candidates: usize,
}

impl Gate {
    fn new(in_dim: usize, n_candidates: usize, rng: &mut LcgRng) -> Self {
        let sc = (1.0 / in_dim as f32).sqrt();
        let w: Vec<f32> = (0..n_candidates * in_dim)
            .map(|_| rng.next_normal() * sc)
            .collect();
        Self {
            w,
            in_dim,
            n_candidates,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let logits = dense(
            x,
            &self.w,
            &vec![0.0_f32; self.n_candidates],
            self.in_dim,
            self.n_candidates,
        );
        softmax(&logits)
    }
}

/// Progressive Layered Extraction (PLE) for multi-task learning.
pub struct Ple {
    pub n_tasks: usize,
    pub n_layers: usize,
    pub expert_dim: usize,
    pub input_dim: usize,
    /// shared_experts[layer]: shared experts for that CGC layer
    shared_experts: Vec<Vec<Expert>>,
    /// task_experts[layer][task]: task-specific experts
    task_experts: Vec<Vec<Vec<Expert>>>,
    /// shared_gates[layer]: gate for each task selecting from shared+task experts
    shared_gates: Vec<Vec<Gate>>,
    /// task_output_w[task]: final linear for each task -> scalar
    task_output_w: Vec<Vec<f32>>,
    task_output_b: Vec<f32>,
}

impl Ple {
    pub fn new(
        n_tasks: usize,
        n_shared_experts: usize,
        n_task_experts: usize,
        expert_dim: usize,
        input_dim: usize,
        n_layers: usize,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if input_dim == 0 || expert_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: input_dim });
        }
        if n_tasks == 0 {
            return Err(RecsysError::Internal {
                msg: "n_tasks must be > 0".into(),
            });
        }

        let mut shared_experts_all = Vec::with_capacity(n_layers);
        let mut task_experts_all = Vec::with_capacity(n_layers);
        let mut shared_gates_all = Vec::with_capacity(n_layers);

        for layer in 0..n_layers {
            let in_dim = if layer == 0 { input_dim } else { expert_dim };

            // Shared experts
            let s_exps: Vec<Expert> = (0..n_shared_experts)
                .map(|_| Expert::new(in_dim, expert_dim, rng))
                .collect();

            // Task-specific experts
            let t_exps: Vec<Vec<Expert>> = (0..n_tasks)
                .map(|_| {
                    (0..n_task_experts)
                        .map(|_| Expert::new(in_dim, expert_dim, rng))
                        .collect()
                })
                .collect();

            // Each task has a gate selecting from its own + shared experts
            let n_candidates = n_task_experts + n_shared_experts;
            let gates: Vec<Gate> = (0..n_tasks)
                .map(|_| Gate::new(in_dim, n_candidates, rng))
                .collect();

            shared_experts_all.push(s_exps);
            task_experts_all.push(t_exps);
            shared_gates_all.push(gates);
        }

        let out_sc = (2.0 / expert_dim as f32).sqrt();
        let task_output_w: Vec<Vec<f32>> = (0..n_tasks)
            .map(|_| {
                (0..expert_dim)
                    .map(|_| rng.next_normal() * out_sc)
                    .collect()
            })
            .collect();
        let task_output_b = vec![0.0_f32; n_tasks];

        Ok(Self {
            n_tasks,
            n_layers,
            expert_dim,
            input_dim,
            shared_experts: shared_experts_all,
            task_experts: task_experts_all,
            shared_gates: shared_gates_all,
            task_output_w,
            task_output_b,
        })
    }

    pub fn forward(&self, x: &[f32]) -> RecsysResult<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(RecsysError::DimensionMismatch {
                expected: self.input_dim,
                got: x.len(),
            });
        }

        // Task-specific representations after each layer
        let mut task_reprs: Vec<Vec<f32>> = (0..self.n_tasks).map(|_| x.to_vec()).collect();

        for layer in 0..self.n_layers {
            let cur_in = if layer == 0 { x } else { &task_reprs[0] };
            let _ = cur_in;

            // Compute shared expert outputs (use x as input for layer 0, else use average task repr)
            let shared_out: Vec<Vec<f32>> = self.shared_experts[layer]
                .iter()
                .map(|e| e.forward(x))
                .collect();

            let mut next_task_reprs = Vec::with_capacity(self.n_tasks);

            for (task, task_repr) in task_reprs.iter().enumerate() {
                let task_in = task_repr;

                // Task-specific expert outputs
                let task_out: Vec<Vec<f32>> = self.task_experts[layer][task]
                    .iter()
                    .map(|e| e.forward(task_in))
                    .collect();

                // Gate: candidates = task_specific + shared
                let gate_weights = self.shared_gates[layer][task].forward(task_in);

                let n_task_exp = task_out.len();
                let d = self.expert_dim;
                let mut mixed = vec![0.0_f32; d];

                for (e_idx, &gw) in gate_weights.iter().enumerate() {
                    let expert_out = if e_idx < n_task_exp {
                        &task_out[e_idx]
                    } else {
                        &shared_out[e_idx - n_task_exp]
                    };
                    for (m, &ev) in mixed.iter_mut().zip(expert_out.iter()) {
                        *m += gw * ev;
                    }
                }

                next_task_reprs.push(mixed);
            }

            task_reprs = next_task_reprs;
        }

        // Final output per task
        let outputs: Vec<f32> = (0..self.n_tasks)
            .map(|task| {
                let repr = &task_reprs[task];
                let logit = self.task_output_b[task]
                    + repr
                        .iter()
                        .zip(self.task_output_w[task].iter())
                        .map(|(&r, &w)| r * w)
                        .sum::<f32>();
                sigmoid(logit)
            })
            .collect();

        Ok(outputs)
    }
}
