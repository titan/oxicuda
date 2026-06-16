//! HyperMAML: meta-learning via hypernetwork-generated fast-adaptation weights.
//!
//! Implements Przewięźlikowski et al. 2022: a hypernetwork (MLP) maps task embeddings
//! to initial parameters for fast adaptation. The hypernetwork weights are meta-learned.

#![allow(clippy::module_name_repetitions)]

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

/// Type alias for a boxed gradient function: `&[f32] -> Vec<f32>`.
type GradFn = Box<dyn Fn(&[f32]) -> Vec<f32>>;

/// Configuration for HyperMAML.
pub struct HyperMamlConfig {
    /// Hidden layer sizes of the hypernetwork MLP (e.g. `[64, 64]` for 2 hidden layers).
    pub hyper_dims: Vec<usize>,
    /// Dimensionality of the task embedding input.
    pub task_emb_dim: usize,
    /// Dimensionality of the output parameter vector to generate.
    pub target_param_dim: usize,
    /// Meta (outer) learning rate.
    pub outer_lr: f32,
    /// Number of inner adaptation steps.
    pub n_inner_steps: usize,
    /// Inner-loop learning rate.
    pub inner_lr: f32,
}

/// HyperMAML learner: hypernetwork generates initial task parameters for fast adaptation.
pub struct HyperMaml {
    /// MLP layer weight matrices, each stored row-major [out_dim × in_dim].
    hyper_weights: Vec<Vec<f32>>,
    /// MLP layer biases, each stored [out_dim].
    hyper_biases: Vec<Vec<f32>>,
    config: HyperMamlConfig,
}

impl HyperMaml {
    /// Build a HyperMAML instance.
    ///
    /// Constructs an MLP:
    ///   `task_emb_dim → hyper_dims[0] → ... → hyper_dims[k-1] → target_param_dim`
    ///
    /// Weights initialized with He initialization: `scale = sqrt(2 / in_dim)`.
    ///
    /// # Errors
    /// - `InvalidEpisodeConfig` if `task_emb_dim == 0`
    /// - `InvalidEpisodeConfig` if `target_param_dim == 0`
    pub fn new(config: HyperMamlConfig, rng: &mut LcgRng) -> MetaResult<Self> {
        if config.task_emb_dim == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "task_emb_dim must be > 0".into(),
            });
        }
        if config.target_param_dim == 0 {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: "target_param_dim must be > 0".into(),
            });
        }

        // Build layer dimension pairs: (in_dim, out_dim)
        let mut layer_dims: Vec<(usize, usize)> = Vec::new();
        let mut prev_dim = config.task_emb_dim;
        for &h in &config.hyper_dims {
            layer_dims.push((prev_dim, h));
            prev_dim = h;
        }
        layer_dims.push((prev_dim, config.target_param_dim));

        let mut hyper_weights = Vec::with_capacity(layer_dims.len());
        let mut hyper_biases = Vec::with_capacity(layer_dims.len());

        for (in_dim, out_dim) in &layer_dims {
            // He initialization: w ~ N(0, sqrt(2/in_dim)) via Box-Muller transform
            let scale = (2.0_f32 / *in_dim as f32).sqrt();
            let n_weights = in_dim * out_dim;
            let mut w = Vec::with_capacity(n_weights);
            let mut count = 0;
            while count < n_weights {
                let u1 = rng.next_f32().max(1e-10); // avoid log(0)
                let u2 = rng.next_f32();
                let r = (-2.0 * u1.ln()).sqrt();
                let angle = 2.0 * core::f32::consts::PI * u2;
                let z0 = r * angle.cos() * scale;
                let z1 = r * angle.sin() * scale;
                w.push(z0);
                count += 1;
                if count < n_weights {
                    w.push(z1);
                    count += 1;
                }
            }
            hyper_weights.push(w);
            // Biases initialized to zero
            hyper_biases.push(vec![0.0_f32; *out_dim]);
        }

        Ok(Self {
            hyper_weights,
            hyper_biases,
            config,
        })
    }

    /// Forward pass through the hypernetwork MLP (ReLU activations, linear final layer).
    ///
    /// Input: `task_emb` of length `task_emb_dim`.
    /// Output: `target_param_dim` parameter vector.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `task_emb.len() != task_emb_dim`
    pub fn generate_params(&self, task_emb: &[f32]) -> MetaResult<Vec<f32>> {
        if task_emb.len() != self.config.task_emb_dim {
            return Err(MetaError::DimensionMismatch {
                expected: self.config.task_emb_dim,
                got: task_emb.len(),
            });
        }

        let n_layers = self.hyper_weights.len();
        let mut current = task_emb.to_vec();

        for layer_idx in 0..n_layers {
            let w = &self.hyper_weights[layer_idx];
            let b = &self.hyper_biases[layer_idx];
            let in_dim = current.len();
            let out_dim = b.len();

            let mut next = vec![0.0_f32; out_dim];
            for (o, (next_o, &bo)) in next.iter_mut().zip(b.iter()).enumerate() {
                let mut val = bo;
                for (wi, &ci) in w[o * in_dim..(o + 1) * in_dim].iter().zip(current.iter()) {
                    val += wi * ci;
                }
                // Apply ReLU on all but the final layer
                if layer_idx < n_layers - 1 {
                    val = val.max(0.0);
                }
                *next_o = val;
            }
            current = next;
        }

        Ok(current)
    }

    /// Generate initial params from hypernetwork, then adapt via inner-loop SGD.
    ///
    /// Runs `n_inner_steps` gradient-descent steps with `inner_lr` using `grad_fn`.
    ///
    /// # Errors
    /// - Propagates `generate_params` errors (e.g., `DimensionMismatch`)
    pub fn adapt_and_eval(
        &self,
        task_emb: &[f32],
        grad_fn: impl Fn(&[f32]) -> Vec<f32>,
    ) -> MetaResult<Vec<f32>> {
        let mut params = self.generate_params(task_emb)?;
        for _ in 0..self.config.n_inner_steps {
            let grad = grad_fn(&params);
            let eff_len = params.len().min(grad.len());
            for (p, &g) in params[..eff_len].iter_mut().zip(grad[..eff_len].iter()) {
                *p -= self.config.inner_lr * g;
            }
        }
        Ok(params)
    }

    /// Meta-update the hypernetwork weights via backpropagation through the MLP.
    ///
    /// For each task `i`:
    /// 1. Forward-pass through hypernetwork to collect activations
    /// 2. Adapt with inner-loop SGD using `eval_grad_fns[i]`
    /// 3. Compute eval gradient `g_eval` at adapted params
    /// 4. Back-propagate through MLP layers to get weight gradients
    /// 5. Update hypernetwork weights
    ///
    /// Returns mean `||g_eval||²` as the proxy meta-loss.
    ///
    /// # Errors
    /// - `EmptySupport` if `task_embs` is empty
    pub fn meta_update(
        &mut self,
        task_embs: &[Vec<f32>],
        eval_grad_fns: &[GradFn],
    ) -> MetaResult<f32> {
        if task_embs.is_empty() {
            return Err(MetaError::EmptySupport);
        }

        let n_tasks = task_embs.len();
        let n_layers = self.hyper_weights.len();

        // Accumulate gradient signals for each layer's weights and biases
        let mut w_grads: Vec<Vec<f32>> = self
            .hyper_weights
            .iter()
            .map(|w| vec![0.0_f32; w.len()])
            .collect();
        let mut b_grads: Vec<Vec<f32>> = self
            .hyper_biases
            .iter()
            .map(|b| vec![0.0_f32; b.len()])
            .collect();

        let mut total_loss = 0.0_f32;

        for (task_idx, emb) in task_embs.iter().enumerate() {
            let grad_fn = eval_grad_fns.get(task_idx).unwrap_or(&eval_grad_fns[0]);

            // Adapt and get the final adapted params
            let adapted = self.adapt_and_eval(emb, grad_fn)?;

            // Eval gradient at adapted params (proxy: 2 * (adapted - target))
            let g_eval = grad_fn(&adapted);
            let sq_norm: f32 = g_eval.iter().map(|&g| g * g).sum();
            total_loss += sq_norm;

            // Signal to back-propagate: d_loss / d_generated_params
            // First-order approximation: d_loss/d_generated ≈ g_eval
            let d_out: Vec<f32> = g_eval.iter().map(|&g| 2.0 * g / n_tasks as f32).collect();

            // Forward pass to collect layer activations for backprop
            let mut activations: Vec<Vec<f32>> = Vec::with_capacity(n_layers + 1);
            activations.push(emb.to_vec());

            for layer_idx in 0..n_layers {
                let w = &self.hyper_weights[layer_idx];
                let b = &self.hyper_biases[layer_idx];
                let in_dim = activations[layer_idx].len();
                let out_dim = b.len();
                let mut next = vec![0.0_f32; out_dim];
                for (o, (next_o, &bo)) in next.iter_mut().zip(b.iter()).enumerate() {
                    let mut val = bo;
                    for (wi, &ci) in w[o * in_dim..(o + 1) * in_dim]
                        .iter()
                        .zip(activations[layer_idx].iter())
                    {
                        val += wi * ci;
                    }
                    if layer_idx < n_layers - 1 {
                        val = val.max(0.0);
                    }
                    *next_o = val;
                }
                activations.push(next);
            }

            // Backward pass through MLP
            let out_dim_last = self.hyper_biases[n_layers - 1].len();
            let mut d_current: Vec<f32> = d_out
                .iter()
                .copied()
                .chain(core::iter::repeat(0.0_f32))
                .take(out_dim_last)
                .collect();

            for layer_idx in (0..n_layers).rev() {
                let in_dim = activations[layer_idx].len();
                let out_dim = self.hyper_biases[layer_idx].len();
                let h_in = &activations[layer_idx].clone();
                let h_out = &activations[layer_idx + 1].clone();

                // Apply ReLU gradient for non-final layers
                let d_pre_act: Vec<f32> = if layer_idx < n_layers - 1 {
                    d_current
                        .iter()
                        .zip(h_out.iter())
                        .map(|(&d, &h)| if h > 0.0 { d } else { 0.0 })
                        .collect()
                } else {
                    d_current.clone()
                };

                // Gradient w.r.t. weights: d_W[o][i] = d_pre_act[o] * h_in[i]
                for (o, &dpa_o) in d_pre_act.iter().enumerate().take(out_dim) {
                    for (i, &h_in_i) in h_in.iter().enumerate().take(in_dim) {
                        w_grads[layer_idx][o * in_dim + i] += dpa_o * h_in_i;
                    }
                }

                // Gradient w.r.t. biases: d_b[o] = d_pre_act[o]
                for (bg, &dpa) in b_grads[layer_idx].iter_mut().zip(d_pre_act.iter()) {
                    *bg += dpa;
                }

                // Gradient w.r.t. input: d_h_in[i] = Σ_o W[o][i] * d_pre_act[o]
                if layer_idx > 0 {
                    let w = &self.hyper_weights[layer_idx];
                    let mut d_in = vec![0.0_f32; in_dim];
                    for (o, &dpa_o) in d_pre_act.iter().enumerate().take(out_dim) {
                        for (i, d_in_i) in d_in.iter_mut().enumerate().take(in_dim) {
                            *d_in_i += w[o * in_dim + i] * dpa_o;
                        }
                    }
                    d_current = d_in;
                }
            }
        }

        // Apply gradient updates to hypernetwork weights
        let outer_lr = self.config.outer_lr;
        for layer_idx in 0..n_layers {
            for (w, &g) in self.hyper_weights[layer_idx]
                .iter_mut()
                .zip(w_grads[layer_idx].iter())
            {
                *w -= outer_lr * g;
            }
            for (b, &g) in self.hyper_biases[layer_idx]
                .iter_mut()
                .zip(b_grads[layer_idx].iter())
            {
                *b -= outer_lr * g;
            }
        }

        let mean_loss = total_loss / n_tasks as f32;
        Ok(mean_loss)
    }

    /// Return the number of hypernetwork layers.
    pub fn hyper_layer_count(&self) -> usize {
        self.hyper_weights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(7)
    }

    fn simple_config() -> HyperMamlConfig {
        HyperMamlConfig {
            hyper_dims: vec![8],
            task_emb_dim: 4,
            target_param_dim: 3,
            outer_lr: 0.001,
            n_inner_steps: 2,
            inner_lr: 0.01,
        }
    }

    fn quadratic_grad(center: f32) -> impl Fn(&[f32]) -> Vec<f32> {
        move |params: &[f32]| params.iter().map(|&p| 2.0 * (p - center)).collect()
    }

    // Test 1: output has target_param_dim elements
    #[test]
    fn generate_params_shape() {
        let mut rng = make_rng();
        let config = simple_config();
        let target_dim = config.target_param_dim;
        let emb_dim = config.task_emb_dim;
        let model = HyperMaml::new(config, &mut rng).expect("new ok");
        let emb = vec![0.1_f32; emb_dim];
        let out = model.generate_params(&emb).expect("generate ok");
        assert_eq!(out.len(), target_dim);
    }

    // Test 2: all outputs are finite
    #[test]
    fn generate_params_finite() {
        let mut rng = make_rng();
        let config = simple_config();
        let emb_dim = config.task_emb_dim;
        let model = HyperMaml::new(config, &mut rng).expect("new ok");
        let emb = vec![0.5_f32; emb_dim];
        let out = model.generate_params(&emb).expect("generate ok");
        for &v in &out {
            assert!(v.is_finite(), "output {v} is not finite");
        }
    }

    // Test 3: adapt changes params vs no-adapt (0 inner steps)
    #[test]
    fn adapt_changes_params() {
        let config_adapt = HyperMamlConfig {
            hyper_dims: vec![8],
            task_emb_dim: 4,
            target_param_dim: 3,
            outer_lr: 0.001,
            n_inner_steps: 3,
            inner_lr: 0.1,
        };
        let config_no_adapt = HyperMamlConfig {
            hyper_dims: vec![8],
            task_emb_dim: 4,
            target_param_dim: 3,
            outer_lr: 0.001,
            n_inner_steps: 0,
            inner_lr: 0.1,
        };

        // Use same seed for both to get identical hypernetwork init
        let mut rng1 = LcgRng::new(77);
        let mut rng2 = LcgRng::new(77);
        let model_adapt = HyperMaml::new(config_adapt, &mut rng1).expect("new ok");
        let model_no = HyperMaml::new(config_no_adapt, &mut rng2).expect("new ok");

        let emb = vec![0.3_f32, 0.5, 0.7, 0.2];
        let grad_fn = quadratic_grad(2.0);

        let adapted = model_adapt
            .adapt_and_eval(&emb, &grad_fn)
            .expect("adapt ok");
        let no_adapt = model_no
            .adapt_and_eval(&emb, &grad_fn)
            .expect("no_adapt ok");

        assert_ne!(adapted, no_adapt, "adapt should change params");
    }

    // Test 4: meta_update returns finite loss
    #[test]
    fn meta_update_finite() {
        let mut rng = make_rng();
        let config = simple_config();
        let emb_dim = config.task_emb_dim;
        let mut model = HyperMaml::new(config, &mut rng).expect("new ok");

        let task_embs = vec![vec![0.1_f32; emb_dim], vec![0.9_f32; emb_dim]];
        let eval_grad_fns: Vec<GradFn> = vec![
            Box::new(quadratic_grad(1.0)),
            Box::new(quadratic_grad(-1.0)),
        ];
        let loss = model
            .meta_update(&task_embs, &eval_grad_fns)
            .expect("meta_update ok");
        assert!(loss.is_finite(), "loss={loss} is not finite");
    }

    // Test 5: empty tasks → Err(EmptySupport)
    #[test]
    fn meta_update_empty_tasks_error() {
        let mut rng = make_rng();
        let config = simple_config();
        let mut model = HyperMaml::new(config, &mut rng).expect("new ok");

        let task_embs: Vec<Vec<f32>> = vec![];
        let eval_grad_fns: Vec<GradFn> = vec![];
        let result = model.meta_update(&task_embs, &eval_grad_fns);
        assert!(matches!(result, Err(MetaError::EmptySupport)));
    }

    // Test 6: zero task_emb_dim → Err
    #[test]
    fn task_emb_dim_zero_error() {
        let mut rng = make_rng();
        let config = HyperMamlConfig {
            hyper_dims: vec![8],
            task_emb_dim: 0,
            target_param_dim: 3,
            outer_lr: 0.001,
            n_inner_steps: 1,
            inner_lr: 0.01,
        };
        let result = HyperMaml::new(config, &mut rng);
        assert!(matches!(
            result,
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    // Test 7: zero target_param_dim → Err
    #[test]
    fn target_param_dim_zero_error() {
        let mut rng = make_rng();
        let config = HyperMamlConfig {
            hyper_dims: vec![8],
            task_emb_dim: 4,
            target_param_dim: 0,
            outer_lr: 0.001,
            n_inner_steps: 1,
            inner_lr: 0.01,
        };
        let result = HyperMaml::new(config, &mut rng);
        assert!(matches!(
            result,
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    // Test 8: hyper_layer_count matches hyper_dims.len() + 1
    #[test]
    fn hyper_layer_count_correct() {
        let mut rng = make_rng();
        let config = HyperMamlConfig {
            hyper_dims: vec![16, 8],
            task_emb_dim: 4,
            target_param_dim: 3,
            outer_lr: 0.001,
            n_inner_steps: 1,
            inner_lr: 0.01,
        };
        // hyper_dims.len() = 2, so total layers = 2 + 1 = 3
        let model = HyperMaml::new(config, &mut rng).expect("new ok");
        assert_eq!(model.hyper_layer_count(), 3);
    }
}
