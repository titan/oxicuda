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
            // Shared experts receive the original input at layer 0; at deeper layers they
            // receive the mean of all task representations from the previous layer.  Using
            // the per-task average preserves the expert_dim dimensionality expected by
            // shared Expert networks at layer > 0 (in_dim = expert_dim there, not input_dim).
            let shared_input: Vec<f32> = if layer == 0 {
                x.to_vec()
            } else {
                let d = self.expert_dim;
                let mut avg = vec![0.0_f32; d];
                for tr in &task_reprs {
                    for (a, &t) in avg.iter_mut().zip(tr.iter()) {
                        *a += t;
                    }
                }
                let n = task_reprs.len() as f32;
                avg.iter_mut().for_each(|v| *v /= n);
                avg
            };

            let shared_out: Vec<Vec<f32>> = self.shared_experts[layer]
                .iter()
                .map(|e| e.forward(&shared_input))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_ple(seed: u64) -> Ple {
        let mut rng = LcgRng::new(seed);
        // 2 tasks, 2 shared experts, 2 task-specific experts, expert_dim=8, input_dim=4, 1 layer
        Ple::new(2, 2, 2, 8, 4, 1, &mut rng).expect("new ok")
    }

    fn sample_input(seed: u64, n: usize) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n).map(|_| rng.next_normal() * 0.5).collect()
    }

    #[test]
    fn output_len_matches_n_tasks() {
        let model = make_ple(1);
        let x = sample_input(10, model.input_dim);
        let out = model.forward(&x).expect("forward ok");
        assert_eq!(out.len(), model.n_tasks);
    }

    #[test]
    fn all_outputs_in_open_unit_interval() {
        let model = make_ple(2);
        let x = sample_input(11, model.input_dim);
        let out = model.forward(&x).expect("forward ok");
        for (i, &v) in out.iter().enumerate() {
            assert!(v > 0.0 && v < 1.0, "task {i} output {v} not in (0,1)");
        }
    }

    #[test]
    fn outputs_are_finite() {
        let model = make_ple(3);
        let x = sample_input(12, model.input_dim);
        let out = model.forward(&x).expect("forward ok");
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "task {i} output {v} is not finite");
        }
    }

    #[test]
    fn determinism_same_seed_same_output() {
        let x = sample_input(99, 4);
        let out1 = make_ple(5).forward(&x).expect("fwd");
        let out2 = make_ple(5).forward(&x).expect("fwd");
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "outputs must be bit-exact");
        }
    }

    #[test]
    fn wrong_input_dim_returns_err() {
        let model = make_ple(6);
        let bad = vec![0.0_f32; model.input_dim + 3];
        assert!(model.forward(&bad).is_err());
    }

    #[test]
    fn invalid_n_tasks_returns_err() {
        let mut rng = LcgRng::new(7);
        assert!(Ple::new(0, 2, 2, 8, 4, 1, &mut rng).is_err());
    }

    #[test]
    fn zero_input_gives_half_probability_for_all_tasks() {
        // Analytical invariant: with zero input vector every expert outputs ReLU(W·0+0)=0,
        // the gated mixture remains 0, and the final logit is 0 (zero bias), so
        // sigmoid(0) = 0.5 exactly for every task.
        let mut rng = LcgRng::new(13);
        let model = Ple::new(3, 2, 2, 8, 4, 1, &mut rng).expect("new ok");
        let zeros = vec![0.0_f32; model.input_dim];
        let out = model.forward(&zeros).expect("fwd ok");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "task {i}: expected 0.5 on zero input, got {v}"
            );
        }
    }

    #[test]
    fn multi_layer_zero_input_also_gives_half() {
        // Verifies the shared-expert fix for layer > 0: the average of zero task
        // representations is still zero, so the zero-input → 0.5 invariant holds
        // regardless of depth.
        let mut rng = LcgRng::new(42);
        let model = Ple::new(2, 2, 1, 8, 4, 3, &mut rng).expect("new ok");
        let zeros = vec![0.0_f32; model.input_dim];
        let out = model.forward(&zeros).expect("fwd ok");
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "task {i}: expected 0.5 on zero input (3 layers), got {v}"
            );
        }
    }
}
