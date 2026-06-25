//! DeepONet: Deep Operator Network for learning operators between function spaces.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// Configuration for DeepONet.
pub struct DeepONetConfig {
    /// Dimensionality of function values at each sensor.
    pub d_input_func: usize,
    /// Number of sensor locations.
    pub n_sensors: usize,
    /// Dimensionality of query coordinates.
    pub d_query: usize,
    /// Width of the output basis (inner product dimension).
    pub p: usize,
    /// Hidden layer widths for the branch net.
    pub branch_hidden: Vec<usize>,
    /// Hidden layer widths for the trunk net.
    pub trunk_hidden: Vec<usize>,
}

/// DeepONet: `G(u)(y) = branch(u) · trunk(y) + bias`.
pub struct DeepONet {
    config: DeepONetConfig,
    branch_w: Vec<Vec<f32>>,
    branch_b: Vec<Vec<f32>>,
    trunk_w: Vec<Vec<f32>>,
    trunk_b: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

fn tanh_vec(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| x.tanh()).collect()
}

fn mlp_forward(
    weights: &[Vec<f32>],
    biases: &[Vec<f32>],
    input: &[f32],
    use_tanh: bool,
) -> PinnResult<Vec<f32>> {
    let mut x = input.to_vec();
    let n_layers = weights.len();
    for (l, (w, b)) in weights.iter().zip(biases.iter()).enumerate() {
        let d_in = x.len();
        let d_out = b.len();
        if w.len() != d_out * d_in {
            return Err(PinnError::DimensionMismatch {
                expected: d_out * d_in,
                got: w.len(),
            });
        }
        let out: Vec<f32> = (0..d_out)
            .map(|i| {
                let dot: f32 = (0..d_in).map(|j| w[i * d_in + j] * x[j]).sum();
                dot + b[i]
            })
            .collect();
        // Apply tanh to all layers except the last
        x = if use_tanh && l < n_layers - 1 {
            tanh_vec(out)
        } else {
            out
        };
    }
    Ok(x)
}

fn init_layer(d_in: usize, d_out: usize, rng: &mut LcgRng) -> (Vec<f32>, Vec<f32>) {
    let scale = (2.0 / d_in as f32).sqrt();
    let w: Vec<f32> = (0..d_out * d_in)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
        .collect();
    let b = vec![0.0_f32; d_out];
    (w, b)
}

impl DeepONet {
    /// Construct a new DeepONet.
    pub fn new(config: DeepONetConfig, rng: &mut LcgRng) -> Self {
        let branch_input = config.n_sensors * config.d_input_func;
        let trunk_input = config.d_query;
        let p = config.p;

        // Build branch layers: branch_input → [branch_hidden...] → p
        let mut branch_layer_sizes = vec![branch_input];
        branch_layer_sizes.extend_from_slice(&config.branch_hidden);
        branch_layer_sizes.push(p);

        let mut branch_w = Vec::new();
        let mut branch_b = Vec::new();
        for win in branch_layer_sizes.windows(2) {
            let (w, b) = init_layer(win[0], win[1], rng);
            branch_w.push(w);
            branch_b.push(b);
        }

        // Build trunk layers: trunk_input → [trunk_hidden...] → p
        let mut trunk_layer_sizes = vec![trunk_input];
        trunk_layer_sizes.extend_from_slice(&config.trunk_hidden);
        trunk_layer_sizes.push(p);

        let mut trunk_w = Vec::new();
        let mut trunk_b = Vec::new();
        for win in trunk_layer_sizes.windows(2) {
            let (w, b) = init_layer(win[0], win[1], rng);
            trunk_w.push(w);
            trunk_b.push(b);
        }

        let bias = vec![0.0_f32; 1]; // scalar bias

        Self {
            config,
            branch_w,
            branch_b,
            trunk_w,
            trunk_b,
            bias,
        }
    }

    /// Branch network: encode function samples `[n_sensors × d_input_func]` → `[p]`.
    pub fn branch_forward(&self, func_samples: &[f32]) -> PinnResult<Vec<f32>> {
        let expected = self.config.n_sensors * self.config.d_input_func;
        if func_samples.len() != expected {
            return Err(PinnError::DimensionMismatch {
                expected,
                got: func_samples.len(),
            });
        }
        mlp_forward(&self.branch_w, &self.branch_b, func_samples, true)
    }

    /// Read-only view of the trunk network's per-layer weight matrices
    /// (row-major `[d_out × d_in]`). Exposed so physics-informed extensions can
    /// propagate dual numbers through the trunk for exact `∂trunk/∂y`.
    #[must_use]
    pub fn trunk_weights(&self) -> &[Vec<f32>] {
        &self.trunk_w
    }

    /// Read-only view of the trunk network's per-layer bias vectors.
    #[must_use]
    pub fn trunk_biases(&self) -> &[Vec<f32>] {
        &self.trunk_b
    }

    /// Trunk network: encode query coordinate `[d_query]` → `[p]`.
    pub fn trunk_forward(&self, query: &[f32]) -> PinnResult<Vec<f32>> {
        if query.len() != self.config.d_query {
            return Err(PinnError::DimensionMismatch {
                expected: self.config.d_query,
                got: query.len(),
            });
        }
        mlp_forward(&self.trunk_w, &self.trunk_b, query, true)
    }

    /// `G(u)(y) = branch(u) · trunk(y) + bias`.
    pub fn forward(&self, func_samples: &[f32], query: &[f32]) -> PinnResult<f32> {
        let b = self.branch_forward(func_samples)?;
        let t = self.trunk_forward(query)?;
        if b.len() != t.len() {
            return Err(PinnError::DimensionMismatch {
                expected: b.len(),
                got: t.len(),
            });
        }
        let dot: f32 = b.iter().zip(t.iter()).map(|(&bi, &ti)| bi * ti).sum();
        Ok(dot + self.bias[0])
    }

    /// Batch forward over multiple queries.
    pub fn forward_batch(
        &self,
        func_samples: &[f32],
        queries: &[f32],
        n_queries: usize,
    ) -> PinnResult<Vec<f32>> {
        let d_q = self.config.d_query;
        if queries.len() != n_queries * d_q {
            return Err(PinnError::DimensionMismatch {
                expected: n_queries * d_q,
                got: queries.len(),
            });
        }
        let branch_out = self.branch_forward(func_samples)?;
        (0..n_queries)
            .map(|i| {
                let q = &queries[i * d_q..(i + 1) * d_q];
                let t = mlp_forward(&self.trunk_w, &self.trunk_b, q, true)?;
                if branch_out.len() != t.len() {
                    return Err(PinnError::DimensionMismatch {
                        expected: branch_out.len(),
                        got: t.len(),
                    });
                }
                let dot: f32 = branch_out
                    .iter()
                    .zip(t.iter())
                    .map(|(&bi, &ti)| bi * ti)
                    .sum();
                Ok(dot + self.bias[0])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> DeepONetConfig {
        DeepONetConfig {
            d_input_func: 1,
            n_sensors: 8,
            d_query: 1,
            p: 16,
            branch_hidden: vec![32],
            trunk_hidden: vec![32],
        }
    }

    #[test]
    fn deeponet_construct() {
        let mut rng = LcgRng::new(1);
        let _model = DeepONet::new(make_config(), &mut rng);
    }

    #[test]
    fn branch_output_shape() {
        let mut rng = LcgRng::new(2);
        let model = DeepONet::new(make_config(), &mut rng);
        let fs = vec![0.1_f32; 8];
        let out = model
            .branch_forward(&fs)
            .expect("branch forward should succeed for valid sensor input");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn trunk_output_shape() {
        let mut rng = LcgRng::new(3);
        let model = DeepONet::new(make_config(), &mut rng);
        let q = vec![0.5_f32];
        let out = model
            .trunk_forward(&q)
            .expect("trunk forward should succeed for valid query input");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn forward_scalar_output() {
        let mut rng = LcgRng::new(4);
        let model = DeepONet::new(make_config(), &mut rng);
        let fs = vec![0.1_f32; 8];
        let q = vec![0.5_f32];
        let out = model
            .forward(&fs, &q)
            .expect("DeepONet forward should produce a finite scalar output");
        assert!(out.is_finite());
    }

    #[test]
    fn forward_batch_shape() {
        let mut rng = LcgRng::new(5);
        let model = DeepONet::new(make_config(), &mut rng);
        let fs = vec![0.1_f32; 8];
        let queries: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let out = model
            .forward_batch(&fs, &queries, 10)
            .expect("batch forward with 10 queries should succeed");
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn forward_batch_all_finite() {
        let mut rng = LcgRng::new(6);
        let model = DeepONet::new(make_config(), &mut rng);
        let fs = vec![0.5_f32; 8];
        let queries = vec![0.3_f32; 5];
        let out = model
            .forward_batch(&fs, &queries, 5)
            .expect("batch forward with 5 queries should produce finite values");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dot_product_manual_check() {
        // With trivial weights, verify the inner product manually
        let cfg = DeepONetConfig {
            d_input_func: 1,
            n_sensors: 2,
            d_query: 1,
            p: 2,
            branch_hidden: vec![],
            trunk_hidden: vec![],
        };
        let mut rng = LcgRng::new(7);
        let model = DeepONet::new(cfg, &mut rng);
        let fs = vec![1.0_f32; 2];
        let q = vec![1.0_f32];
        let b = model
            .branch_forward(&fs)
            .expect("branch forward should succeed for valid sensor input in dot product check");
        let t = model
            .trunk_forward(&q)
            .expect("trunk forward should succeed for valid query in dot product check");
        let manual_dot: f32 = b
            .iter()
            .zip(t.iter())
            .map(|(&bi, &ti)| bi * ti)
            .sum::<f32>()
            + model.bias[0];
        let model_out = model
            .forward(&fs, &q)
            .expect("DeepONet forward should match manual dot product computation");
        assert!((manual_dot - model_out).abs() < 1e-5);
    }

    #[test]
    fn branch_dim_mismatch_error() {
        let mut rng = LcgRng::new(8);
        let model = DeepONet::new(make_config(), &mut rng);
        let result = model.branch_forward(&[0.0; 5]); // expects 8
        assert!(result.is_err());
    }
}
