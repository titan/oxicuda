//! DCN V2: Deep & Cross Network with Low-Rank Cross Layers.
//!
//! Reference: Wang et al. "DCN V2: Improved Deep & Cross Network and Practical Lessons for
//! Web-scale Learning to Rank Systems", WWW 2021.
//!
//! Cross network: explicit polynomial feature interactions via `x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l`.
//! Low-rank cross: `W_l ≈ U_l V_l^T` where `U_l, V_l ∈ R^{d×r}` — saves O(d²) → O(dr) memory.
//! Deep network: standard ReLU-MLP in parallel or series with the cross network.
//! Output head: linear layer over the concatenated (or stacked) representations.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── DcnV2Mode ────────────────────────────────────────────────────────────────

/// How the cross network and deep network are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcnV2Mode {
    /// Parallel: cross and deep run independently; outputs concatenated before head.
    Parallel,
    /// Stacked: cross network output is fed as input to the deep network; then
    /// cross and deep outputs are concatenated before head.
    Stacked,
}

// ─── DcnV2Config ──────────────────────────────────────────────────────────────

/// Configuration for `DcnV2`.
#[derive(Debug, Clone)]
pub struct DcnV2Config {
    /// Input feature dimension d (after embedding/preprocessing).
    pub input_dim: usize,
    /// Number of cross layers L (default 3).
    pub n_cross_layers: usize,
    /// Deep network hidden dimension (default 256).
    pub deep_hidden: usize,
    /// Number of deep network layers (default 3).
    pub n_deep_layers: usize,
    /// Low-rank dimension r for cross layer. `None` = full-rank W (d×d).
    pub low_rank: Option<usize>,
    /// Combination mode (default Parallel).
    pub mode: DcnV2Mode,
    /// Number of output classes (1 for regression).
    pub n_classes: usize,
}

impl Default for DcnV2Config {
    fn default() -> Self {
        Self {
            input_dim: 64,
            n_cross_layers: 3,
            deep_hidden: 256,
            n_deep_layers: 3,
            low_rank: Some(32),
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        }
    }
}

// ─── CrossLayerWeights ────────────────────────────────────────────────────────

/// Weight bundle for a single cross layer.
#[derive(Debug, Clone)]
pub struct CrossLayerWeights {
    /// Full-rank: W ∈ R^{d×d} stored as `[d×d]` row-major.
    /// Low-rank: U ∈ R^{d×r} stored as `[d×r]` row-major.
    pub w_or_u: Vec<f32>,
    /// Low-rank only: V ∈ R^{d×r} stored as `[d×r]` row-major (empty if full-rank).
    pub v: Vec<f32>,
    /// Bias b ∈ R^d.
    pub b: Vec<f32>,
}

// ─── DeepLayerWeights ─────────────────────────────────────────────────────────

/// Weight bundle for a single deep MLP layer.
#[derive(Debug, Clone)]
pub struct DeepLayerWeights {
    /// W ∈ R^{out × in}, stored row-major.
    pub w: Vec<f32>,
    /// b ∈ R^{out}.
    pub b: Vec<f32>,
}

// ─── DcnV2Weights ─────────────────────────────────────────────────────────────

/// Full DCN V2 weight bundle.
#[derive(Debug, Clone)]
pub struct DcnV2Weights {
    /// Cross network layers: `n_cross_layers` items.
    pub cross_layers: Vec<CrossLayerWeights>,
    /// Deep network layers: `n_deep_layers` items.
    /// Layer 0: `[deep_hidden × input_dim]`; layers 1+: `[deep_hidden × deep_hidden]`.
    pub deep_layers: Vec<DeepLayerWeights>,
    /// Output head weight `[n_classes × (input_dim + deep_hidden)]`.
    pub head_w: Vec<f32>,
    /// Output head bias `[n_classes]`.
    pub head_b: Vec<f32>,
}

impl DcnV2Weights {
    /// Create randomly initialised weights for a DCN V2 model.
    ///
    /// Kaiming-uniform init: U(-k, k) where k = sqrt(6 / fan_in). Biases = zeros.
    pub fn new_random(cfg: &DcnV2Config, rng: &mut LcgRng) -> Self {
        let d = cfg.input_dim;
        let r_opt = cfg.low_rank;

        // ── Cross layers ──────────────────────────────────────────────────────
        let cross_layers: Vec<CrossLayerWeights> = (0..cfg.n_cross_layers)
            .map(|_| {
                if let Some(r) = r_opt {
                    // Low-rank: U ∈ R^{d×r}, V ∈ R^{d×r}
                    let k_u = (6.0_f32 / d as f32).sqrt();
                    let w_or_u: Vec<f32> = (0..d * r)
                        .map(|_| rng.next_f32() * 2.0 * k_u - k_u)
                        .collect();
                    let k_v = (6.0_f32 / r as f32).sqrt();
                    let v: Vec<f32> = (0..d * r)
                        .map(|_| rng.next_f32() * 2.0 * k_v - k_v)
                        .collect();
                    CrossLayerWeights {
                        w_or_u,
                        v,
                        b: vec![0.0_f32; d],
                    }
                } else {
                    // Full-rank: W ∈ R^{d×d}
                    let k = (6.0_f32 / d as f32).sqrt();
                    let w_or_u: Vec<f32> =
                        (0..d * d).map(|_| rng.next_f32() * 2.0 * k - k).collect();
                    CrossLayerWeights {
                        w_or_u,
                        v: Vec::new(),
                        b: vec![0.0_f32; d],
                    }
                }
            })
            .collect();

        // ── Deep layers ───────────────────────────────────────────────────────
        let h = cfg.deep_hidden;
        let n = cfg.n_deep_layers;
        let deep_layers: Vec<DeepLayerWeights> = (0..n)
            .map(|layer_idx| {
                let in_dim = if layer_idx == 0 { d } else { h };
                let k = (6.0_f32 / in_dim as f32).sqrt();
                let w: Vec<f32> = (0..h * in_dim)
                    .map(|_| rng.next_f32() * 2.0 * k - k)
                    .collect();
                DeepLayerWeights {
                    w,
                    b: vec![0.0_f32; h],
                }
            })
            .collect();

        // ── Head ─────────────────────────────────────────────────────────────
        let head_in = d + h; // always concat cross + deep
        let k_head = (6.0_f32 / head_in as f32).sqrt();
        let head_w: Vec<f32> = (0..cfg.n_classes * head_in)
            .map(|_| rng.next_f32() * 2.0 * k_head - k_head)
            .collect();
        let head_b = vec![0.0_f32; cfg.n_classes];

        Self {
            cross_layers,
            deep_layers,
            head_w,
            head_b,
        }
    }
}

// ─── DcnV2 ───────────────────────────────────────────────────────────────────

/// DCN V2 model (inference only).
pub struct DcnV2 {
    /// Model configuration.
    pub config: DcnV2Config,
}

impl DcnV2 {
    /// Construct a new `DcnV2` instance, validating the configuration.
    pub fn new(config: DcnV2Config) -> TabularResult<Self> {
        if config.input_dim == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if let Some(r) = config.low_rank {
            if r == 0 {
                return Err(TabularError::InvalidEmbedDim { dim: 0 });
            }
            if r > config.input_dim {
                return Err(TabularError::InvalidAttentionDim {
                    dim: config.input_dim,
                });
            }
        }
        Ok(Self { config })
    }

    /// Apply one cross layer.
    ///
    /// **Full-rank**: `x_{l+1} = x_0 ⊙ (W x_l + b) + x_l`
    ///
    /// **Low-rank** (`W = U V^T`): `x_{l+1} = x_0 ⊙ (U (V^T x_l) + b) + x_l`
    ///
    /// V is stored as `[d × r]` row-major; `V^T x_l` is computed column-wise.
    pub fn cross_layer(&self, x0: &[f32], xl: &[f32], weights: &CrossLayerWeights) -> Vec<f32> {
        let d = self.config.input_dim;

        // Compute W x_l (or U V^T x_l)
        let wx: Vec<f32> = if let Some(r) = self.config.low_rank {
            // V^T x_l: [r] vector.  V stored as [d × r] row-major.
            // (V^T)[j, i] = V[i*r + j]  →  vx[j] = sum_i V[i*r+j] * xl[i]
            let mut vx = vec![0.0_f32; r];
            for (j, vxj) in vx.iter_mut().enumerate() {
                let mut acc = 0.0_f32;
                for (i, &xli) in xl.iter().enumerate() {
                    acc += weights.v[i * r + j] * xli;
                }
                *vxj = acc;
            }
            // U V^T x_l: U is [d × r] row-major.  UVx[i] = sum_j U[i*r+j] * vx[j]
            (0..d)
                .map(|i| {
                    let mut acc = 0.0_f32;
                    for (j, &vxj) in vx.iter().enumerate() {
                        acc += weights.w_or_u[i * r + j] * vxj;
                    }
                    acc
                })
                .collect()
        } else {
            // Full-rank: W [d×d] @ xl [d] → [d]
            (0..d)
                .map(|i| {
                    let mut acc = 0.0_f32;
                    for (k, &xlk) in xl.iter().enumerate() {
                        acc += weights.w_or_u[i * d + k] * xlk;
                    }
                    acc
                })
                .collect()
        };

        // x_{l+1} = x_0 ⊙ (Wx + b) + xl
        (0..d)
            .map(|i| x0[i] * (wx[i] + weights.b[i]) + xl[i])
            .collect()
    }

    /// Apply L cross layers from input `x0`.
    ///
    /// Returns final cross output `[input_dim]`.
    /// If `n_cross_layers == 0`, returns `x0.to_vec()` (identity).
    pub fn cross_network(&self, x0: &[f32], weights: &DcnV2Weights) -> Vec<f32> {
        if weights.cross_layers.is_empty() {
            return x0.to_vec();
        }
        let mut xl = x0.to_vec();
        for layer_w in &weights.cross_layers {
            xl = self.cross_layer(x0, &xl, layer_w);
        }
        xl
    }

    /// Apply the deep MLP with ReLU activations.
    ///
    /// - Input: `[input_dim]` (or `[input_dim]` when stacked mode passes cross output).
    /// - Returns: `[deep_hidden]`.
    pub fn deep_network(&self, x: &[f32], weights: &DcnV2Weights) -> TabularResult<Vec<f32>> {
        let d = self.config.input_dim;
        let h = self.config.deep_hidden;

        if x.len() != d {
            return Err(TabularError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }

        if weights.deep_layers.is_empty() {
            // No deep layers: project input to deep_hidden directly (zero init fallback)
            return Ok(vec![0.0_f32; h]);
        }

        let mut hidden: Vec<f32> = x.to_vec();
        for (layer_idx, layer_w) in weights.deep_layers.iter().enumerate() {
            let in_dim = if layer_idx == 0 { d } else { h };
            let mut new_h = layer_w.b.clone();
            for (o, nh) in new_h.iter_mut().enumerate() {
                let mut acc = 0.0_f32;
                for (i, &hi) in hidden.iter().enumerate() {
                    acc += layer_w.w[o * in_dim + i] * hi;
                }
                *nh += acc;
            }
            // ReLU
            for v in &mut new_h {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            hidden = new_h;
        }
        Ok(hidden)
    }

    /// Forward pass on one sample.
    ///
    /// - `x`: `[input_dim]` → logits: `[n_classes]`.
    ///
    /// In **Parallel** mode: cross and deep run on `x` independently, then concat.
    /// In **Stacked** mode: cross output is fed to deep; then concat cross + deep outputs.
    pub fn forward_single(&self, x: &[f32], weights: &DcnV2Weights) -> TabularResult<Vec<f32>> {
        let d = self.config.input_dim;

        if x.len() != d {
            return Err(TabularError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }

        let (cross_out, deep_out) = match self.config.mode {
            DcnV2Mode::Parallel => {
                let cross_out = self.cross_network(x, weights);
                let deep_out = self.deep_network(x, weights)?;
                (cross_out, deep_out)
            }
            DcnV2Mode::Stacked => {
                let cross_out = self.cross_network(x, weights);
                // Feed cross output into deep
                let deep_out = self.deep_network(&cross_out, weights)?;
                (cross_out, deep_out)
            }
        };

        // Concatenate cross + deep
        let mut concat = Vec::with_capacity(cross_out.len() + deep_out.len());
        concat.extend_from_slice(&cross_out);
        concat.extend_from_slice(&deep_out);

        // Linear head
        let head_in = concat.len();
        let mut logits = weights.head_b.clone();
        for (c, logit) in logits.iter_mut().enumerate() {
            for (i, &cv) in concat.iter().enumerate() {
                *logit += weights.head_w[c * head_in + i] * cv;
            }
        }
        Ok(logits)
    }

    /// Forward pass on a batch.
    ///
    /// - `x`: `[n_samples × input_dim]` → logits: `[n_samples × n_classes]`.
    pub fn forward(&self, x: &[f32], weights: &DcnV2Weights) -> TabularResult<Vec<f32>> {
        let d = self.config.input_dim;
        let n_classes = self.config.n_classes;

        if !x.len().is_multiple_of(d) {
            return Err(TabularError::DimensionMismatch {
                expected: (x.len() / d) * d,
                got: x.len(),
            });
        }
        let n_samples = x.len() / d;

        let mut all_logits = Vec::with_capacity(n_samples * n_classes);
        for s in 0..n_samples {
            let row = &x[s * d..(s + 1) * d];
            let logits = self.forward_single(row, weights)?;
            all_logits.extend_from_slice(&logits);
        }
        Ok(all_logits)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── DcnV2::new validation ─────────────────────────────────────────────────

    #[test]
    fn dcn_v2_new_input_dim_zero_is_err() {
        let cfg = DcnV2Config {
            input_dim: 0,
            ..DcnV2Config::default()
        };
        assert!(DcnV2::new(cfg).is_err());
    }

    #[test]
    fn dcn_v2_new_low_rank_zero_is_err() {
        let cfg = DcnV2Config {
            input_dim: 8,
            low_rank: Some(0),
            ..DcnV2Config::default()
        };
        assert!(DcnV2::new(cfg).is_err());
    }

    #[test]
    fn dcn_v2_new_low_rank_exceeds_input_dim_is_err() {
        let cfg = DcnV2Config {
            input_dim: 8,
            low_rank: Some(16), // 16 > 8
            ..DcnV2Config::default()
        };
        assert!(DcnV2::new(cfg).is_err());
    }

    // ── cross_layer algebraic tests ───────────────────────────────────────────

    #[test]
    fn cross_layer_full_rank_identity_weight() {
        // W = I (identity), b = 0: x_{l+1} = x_0 ⊙ xl + xl
        let d = 4;
        let cfg = DcnV2Config {
            input_dim: d,
            n_cross_layers: 1,
            deep_hidden: 4,
            n_deep_layers: 1,
            low_rank: None, // full-rank
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg).unwrap();

        // Build identity W
        let mut w = vec![0.0_f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        let weights = CrossLayerWeights {
            w_or_u: w,
            v: Vec::new(),
            b: vec![0.0_f32; d],
        };

        let x0 = vec![1.0_f32, 2.0, 3.0, 4.0];
        let xl = vec![2.0_f32, 1.0, 0.5, 0.25];

        let out = model.cross_layer(&x0, &xl, &weights);
        // x_{l+1}[i] = x0[i] * xl[i] + xl[i]
        let expected: Vec<f32> = x0.iter().zip(xl.iter()).map(|(&a, &b)| a * b + b).collect();
        for (i, (&o, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((o - e).abs() < 1e-5, "index {i}: expected {e}, got {o}");
        }
    }

    #[test]
    fn cross_layer_zero_weight_pure_residual() {
        // W = 0, b = 0: x_{l+1} = x_0 ⊙ 0 + xl = xl (pure residual)
        let d = 3;
        let cfg = DcnV2Config {
            input_dim: d,
            n_cross_layers: 1,
            deep_hidden: 4,
            n_deep_layers: 1,
            low_rank: None,
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg).unwrap();
        let weights = CrossLayerWeights {
            w_or_u: vec![0.0_f32; d * d],
            v: Vec::new(),
            b: vec![0.0_f32; d],
        };
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let xl = vec![4.0_f32, 5.0, 6.0];
        let out = model.cross_layer(&x0, &xl, &weights);
        for (i, (&o, &e)) in out.iter().zip(xl.iter()).enumerate() {
            assert!(
                (o - e).abs() < 1e-5,
                "index {i}: expected {e} (residual), got {o}"
            );
        }
    }

    #[test]
    fn cross_layer_x0_zeros_pure_residual() {
        // x0 = 0: x_{l+1} = 0 ⊙ (Wx_l + b) + xl = xl
        let d = 4;
        let cfg = DcnV2Config {
            input_dim: d,
            n_cross_layers: 1,
            deep_hidden: 4,
            n_deep_layers: 1,
            low_rank: None,
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg).unwrap();
        let mut rng = LcgRng::new(7);
        // Random W and b
        let w: Vec<f32> = (0..d * d).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
        let weights = CrossLayerWeights {
            w_or_u: w,
            v: Vec::new(),
            b,
        };
        let x0 = vec![0.0_f32; d];
        let xl = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = model.cross_layer(&x0, &xl, &weights);
        for (i, (&o, &e)) in out.iter().zip(xl.iter()).enumerate() {
            assert!(
                (o - e).abs() < 1e-5,
                "index {i}: expected {e} when x0=0, got {o}"
            );
        }
    }

    // ── cross_network tests ───────────────────────────────────────────────────

    #[test]
    fn cross_network_output_shape() {
        let d = 16;
        let cfg = DcnV2Config {
            input_dim: d,
            n_cross_layers: 3,
            deep_hidden: 32,
            n_deep_layers: 2,
            low_rank: Some(4),
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(1);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x0: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let out = model.cross_network(&x0, &weights);
        assert_eq!(out.len(), d);
    }

    #[test]
    fn cross_network_zero_layers_identity() {
        let d = 8;
        let cfg = DcnV2Config {
            input_dim: d,
            n_cross_layers: 0,
            deep_hidden: 16,
            n_deep_layers: 1,
            low_rank: None,
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(2);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x0 = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = model.cross_network(&x0, &weights);
        assert_eq!(out, x0, "zero cross layers must be identity");
    }

    // ── deep_network tests ────────────────────────────────────────────────────

    #[test]
    fn deep_network_output_shape() {
        let d = 16;
        let h = 32;
        let cfg = DcnV2Config {
            input_dim: d,
            n_cross_layers: 2,
            deep_hidden: h,
            n_deep_layers: 2,
            low_rank: Some(4),
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(3);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.05).collect();
        let out = model.deep_network(&x, &weights).unwrap();
        assert_eq!(out.len(), h, "deep_network output shape mismatch");
    }

    #[test]
    fn deep_network_output_is_finite() {
        let d = 32;
        let cfg = DcnV2Config {
            input_dim: d,
            n_cross_layers: 2,
            deep_hidden: 64,
            n_deep_layers: 3,
            low_rank: Some(8),
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(42);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x: Vec<f32> = (0..d).map(|i| (i as f32).sin()).collect();
        let out = model.deep_network(&x, &weights).unwrap();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "deep_network output must be finite"
        );
    }

    // ── forward_single tests ──────────────────────────────────────────────────

    #[test]
    fn forward_single_output_shape() {
        let cfg = DcnV2Config {
            input_dim: 16,
            n_cross_layers: 2,
            deep_hidden: 32,
            n_deep_layers: 2,
            low_rank: Some(4),
            mode: DcnV2Mode::Parallel,
            n_classes: 3,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(5);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x = vec![0.1_f32; 16];
        let logits = model.forward_single(&x, &weights).unwrap();
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn forward_single_parallel_mode() {
        let cfg = DcnV2Config {
            input_dim: 8,
            n_cross_layers: 2,
            deep_hidden: 16,
            n_deep_layers: 2,
            low_rank: Some(2),
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(99);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x = vec![0.5_f32; 8];
        let logits = model.forward_single(&x, &weights).unwrap();
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_single_stacked_mode() {
        let cfg = DcnV2Config {
            input_dim: 8,
            n_cross_layers: 2,
            deep_hidden: 16,
            n_deep_layers: 2,
            low_rank: Some(2),
            mode: DcnV2Mode::Stacked,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(13);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x = vec![0.3_f32; 8];
        let logits = model.forward_single(&x, &weights).unwrap();
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    // ── forward (batch) tests ─────────────────────────────────────────────────

    #[test]
    fn forward_batch_output_shape() {
        let cfg = DcnV2Config {
            input_dim: 16,
            n_cross_layers: 2,
            deep_hidden: 32,
            n_deep_layers: 2,
            low_rank: Some(4),
            mode: DcnV2Mode::Parallel,
            n_classes: 3,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(7);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x = vec![0.1_f32; 5 * 16]; // 5 samples
        let logits = model.forward(&x, &weights).unwrap();
        assert_eq!(logits.len(), 5 * 3);
    }

    #[test]
    fn forward_batch_output_is_finite() {
        let cfg = DcnV2Config {
            input_dim: 16,
            n_cross_layers: 3,
            deep_hidden: 32,
            n_deep_layers: 3,
            low_rank: Some(4),
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(11);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x: Vec<f32> = (0..4 * 16).map(|i| (i as f32) * 0.01).collect();
        let logits = model.forward(&x, &weights).unwrap();
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "batch logits must be finite"
        );
    }

    // ── DcnV2Weights initialisation test ─────────────────────────────────────

    #[test]
    fn dcn_v2_weights_new_random_finite() {
        let cfg = DcnV2Config::default();
        let mut rng = LcgRng::new(33);
        let w = DcnV2Weights::new_random(&cfg, &mut rng);
        for layer in &w.cross_layers {
            assert!(layer.w_or_u.iter().all(|v| v.is_finite()));
            assert!(layer.v.iter().all(|v| v.is_finite()));
            assert!(layer.b.iter().all(|v| v.is_finite()));
        }
        for layer in &w.deep_layers {
            assert!(layer.w.iter().all(|v| v.is_finite()));
            assert!(layer.b.iter().all(|v| v.is_finite()));
        }
        assert!(w.head_w.iter().all(|v| v.is_finite()));
        assert!(w.head_b.iter().all(|v| v.is_finite()));
    }

    // ── n_classes=1 regression ────────────────────────────────────────────────

    #[test]
    fn forward_regression_n_classes_1() {
        let cfg = DcnV2Config {
            input_dim: 8,
            n_cross_layers: 2,
            deep_hidden: 16,
            n_deep_layers: 2,
            low_rank: Some(2),
            mode: DcnV2Mode::Parallel,
            n_classes: 1,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(17);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x = vec![0.5_f32; 8];
        let logits = model.forward_single(&x, &weights).unwrap();
        assert_eq!(logits.len(), 1);
    }

    // ── full-rank vs low-rank same shape ─────────────────────────────────────

    #[test]
    fn full_rank_cross_layer_produces_correct_shape() {
        let d = 6;
        let cfg = DcnV2Config {
            input_dim: d,
            n_cross_layers: 2,
            deep_hidden: 12,
            n_deep_layers: 1,
            low_rank: None, // full-rank
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model = DcnV2::new(cfg.clone()).unwrap();
        let mut rng = LcgRng::new(22);
        let weights = DcnV2Weights::new_random(&cfg, &mut rng);
        let x0 = vec![0.1_f32; d];
        let out = model.cross_network(&x0, &weights);
        assert_eq!(out.len(), d, "full-rank cross_network output shape");
    }

    #[test]
    fn low_rank_cross_layer_same_output_shape_as_full_rank() {
        let d = 8;
        // low-rank
        let cfg_lr = DcnV2Config {
            input_dim: d,
            n_cross_layers: 2,
            deep_hidden: 16,
            n_deep_layers: 1,
            low_rank: Some(2),
            mode: DcnV2Mode::Parallel,
            n_classes: 2,
        };
        let model_lr = DcnV2::new(cfg_lr.clone()).unwrap();
        let mut rng = LcgRng::new(44);
        let w_lr = DcnV2Weights::new_random(&cfg_lr, &mut rng);
        let x0 = vec![0.5_f32; d];
        let out_lr = model_lr.cross_network(&x0, &w_lr);
        assert_eq!(out_lr.len(), d);

        // full-rank
        let cfg_fr = DcnV2Config {
            low_rank: None,
            ..cfg_lr
        };
        let model_fr = DcnV2::new(cfg_fr.clone()).unwrap();
        let mut rng2 = LcgRng::new(44);
        let w_fr = DcnV2Weights::new_random(&cfg_fr, &mut rng2);
        let out_fr = model_fr.cross_network(&x0, &w_fr);
        assert_eq!(out_fr.len(), d);
    }
}
