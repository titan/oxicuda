//! TabNet: Attentive Interpretable Tabular Learning (Arik & Pfister 2021).
//!
//! Architecture:
//! - Shared feature transformer (FC → BN → GLU) shared across all steps.
//! - Step-specific transformer (FC → BN → GLU) per step.
//! - Attention: `M_i = sparsemax(P_i * BN(W_att * h))` where `P_i` penalises reuse.
//! - Feature selection: `h_selected = M_i * features`.
//! - Step output: `h_i = ReLU(step_transform(shared_transform(h_selected)))`.
//! - Final: `FC(mean(h_i))` → logits.

use super::sparsemax::sparsemax;
use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── GLU ─────────────────────────────────────────────────────────────────────

/// Gated Linear Unit: `GLU(x) = x[:d/2] ⊙ σ(x[d/2:])`.
///
/// The input must have even length; returns a vector of length `input.len() / 2`.
pub fn glu(x: &[f32]) -> TabularResult<Vec<f32>> {
    let n = x.len();
    if !n.is_multiple_of(2) {
        return Err(TabularError::DimensionMismatch {
            expected: n + 1,
            got: n,
        });
    }
    let half = n / 2;
    let out = (0..half)
        .map(|i| {
            let val = x[i];
            let gate = x[i + half];
            val * sigmoid(gate)
        })
        .collect();
    Ok(out)
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─── BatchNorm1d ─────────────────────────────────────────────────────────────

/// Simple 1-D batch normalisation with learnable γ/β parameters.
pub struct BatchNorm1d {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub eps: f32,
}

impl BatchNorm1d {
    /// Initialise with `gamma = 1`, `beta = 0`.
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0_f32; dim],
            beta: vec![0.0_f32; dim],
            eps: 1e-5,
        }
    }

    /// Inference-time normalisation using pre-computed `mean` and `var` vectors.
    ///
    /// Input `x`: flat `[dim]` vector.  Returns normalised `[dim]` vector.
    pub fn forward_inference(
        &self,
        x: &[f32],
        mean: &[f32],
        var: &[f32],
    ) -> TabularResult<Vec<f32>> {
        let dim = self.gamma.len();
        if x.len() != dim {
            return Err(TabularError::DimensionMismatch {
                expected: dim,
                got: x.len(),
            });
        }
        let out = (0..dim)
            .map(|i| {
                let norm = (x[i] - mean[i]) / (var[i] + self.eps).sqrt();
                norm * self.gamma[i] + self.beta[i]
            })
            .collect();
        Ok(out)
    }

    /// Compute batch statistics and return `(normalised, mean, var)`.
    ///
    /// `x` is a flat `[batch_size * dim]` row-major matrix.
    pub fn normalize_batch(
        &self,
        x: &[f32],
        batch_size: usize,
    ) -> TabularResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let dim = self.gamma.len();
        if x.len() != batch_size * dim {
            return Err(TabularError::DimensionMismatch {
                expected: batch_size * dim,
                got: x.len(),
            });
        }
        if batch_size == 0 {
            return Err(TabularError::InsufficientSamples { need: 1, got: 0 });
        }
        let n = batch_size as f32;

        // Compute mean per feature
        let mut mean = vec![0.0_f32; dim];
        for row in 0..batch_size {
            for col in 0..dim {
                mean[col] += x[row * dim + col];
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        // Compute variance per feature
        let mut var = vec![0.0_f32; dim];
        for row in 0..batch_size {
            for col in 0..dim {
                let diff = x[row * dim + col] - mean[col];
                var[col] += diff * diff;
            }
        }
        for v in &mut var {
            *v /= n;
        }

        // Normalise
        let mut normed = vec![0.0_f32; batch_size * dim];
        for row in 0..batch_size {
            for col in 0..dim {
                let norm = (x[row * dim + col] - mean[col]) / (var[col] + self.eps).sqrt();
                normed[row * dim + col] = norm * self.gamma[col] + self.beta[col];
            }
        }

        Ok((normed, mean, var))
    }
}

// ─── TabNetConfig ─────────────────────────────────────────────────────────────

/// Configuration for `TabNetLayer`.
pub struct TabNetConfig {
    /// Number of input features.
    pub n_features: usize,
    /// Dimension of step output (prediction layer).
    pub n_d: usize,
    /// Attention embedding dimension (often `n_d`).
    pub n_a: usize,
    /// Number of sequential attention steps (3–10).
    pub n_steps: usize,
    /// Feature-reuse penalisation coefficient γ (1.0–2.0).
    pub gamma: f32,
    /// Output dimension (1 for regression, `n_classes` for classification).
    pub n_classes: usize,
}

// ─── TabNetLayer ─────────────────────────────────────────────────────────────

/// Xavier-initialised fully-connected layer weight helper.
fn xavier_init(rng: &mut LcgRng, fan_in: usize, fan_out: usize, n: usize) -> Vec<f32> {
    let std_dev = (2.0_f32 / (fan_in + fan_out) as f32).sqrt();
    let mut w = vec![0.0_f32; n];
    rng.fill_normal_scaled(&mut w, std_dev);
    w
}

/// TabNet layer with shared + step-specific transformers and attention.
pub struct TabNetLayer {
    // Shared FC: maps [n_features] → [2*(n_d + n_a)]
    shared_w: Vec<f32>,
    shared_b: Vec<f32>,
    // Per-step FC: maps [(n_d + n_a)] → [2*(n_d + n_a)]
    step_w: Vec<Vec<f32>>,
    step_b: Vec<Vec<f32>>,
    // Per-step attention FC: maps [(n_d + n_a)] → [n_features]
    att_w: Vec<Vec<f32>>,
    att_b: Vec<Vec<f32>>,
    // Output head: maps [n_d] → [n_classes]
    final_w: Vec<f32>,
    final_b: Vec<f32>,
    bn: BatchNorm1d,
    config: TabNetConfig,
}

impl TabNetLayer {
    /// Construct a new `TabNetLayer` with Xavier weight initialisation.
    pub fn new(cfg: TabNetConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        if cfg.n_features == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if cfg.n_steps == 0 {
            return Err(TabularError::InvalidStepCount { steps: 0 });
        }
        if cfg.n_d == 0 || cfg.n_a == 0 {
            return Err(TabularError::InvalidAttentionDim { dim: 0 });
        }

        let na_nd = cfg.n_a + cfg.n_d;
        let out_shared = 2 * na_nd;

        let shared_w = xavier_init(rng, cfg.n_features, out_shared, cfg.n_features * out_shared);
        let shared_b = vec![0.0_f32; out_shared];

        let mut step_w = Vec::with_capacity(cfg.n_steps);
        let mut step_b = Vec::with_capacity(cfg.n_steps);
        let mut att_w = Vec::with_capacity(cfg.n_steps);
        let mut att_b = Vec::with_capacity(cfg.n_steps);

        for _ in 0..cfg.n_steps {
            step_w.push(xavier_init(rng, na_nd, out_shared, na_nd * out_shared));
            step_b.push(vec![0.0_f32; out_shared]);
            att_w.push(xavier_init(
                rng,
                na_nd,
                cfg.n_features,
                na_nd * cfg.n_features,
            ));
            att_b.push(vec![0.0_f32; cfg.n_features]);
        }

        let final_w = xavier_init(rng, cfg.n_d, cfg.n_classes, cfg.n_d * cfg.n_classes);
        let final_b = vec![0.0_f32; cfg.n_classes];

        let bn = BatchNorm1d::new(cfg.n_features);

        Ok(Self {
            shared_w,
            shared_b,
            step_w,
            step_b,
            att_w,
            att_b,
            final_w,
            final_b,
            bn,
            config: cfg,
        })
    }

    /// Matrix-vector multiply: `y = W x + b` where `W` is row-major `[out * in_dim]`.
    fn matvec(w: &[f32], b: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut y = b.to_vec();
        for o in 0..out_dim {
            for i in 0..in_dim {
                y[o] += w[o * in_dim + i] * x[i];
            }
        }
        y
    }

    /// Forward pass for a **single** sample.
    ///
    /// Returns `(logits [n_classes], attention_masks [n_steps * n_features])`.
    pub fn forward(&self, x: &[f32]) -> TabularResult<(Vec<f32>, Vec<f32>)> {
        let cfg = &self.config;
        if x.len() != cfg.n_features {
            return Err(TabularError::DimensionMismatch {
                expected: cfg.n_features,
                got: x.len(),
            });
        }

        let na_nd = cfg.n_a + cfg.n_d;

        // Use trivial BN stats (mean=0, var=1) for single-sample inference
        let bn_mean = vec![0.0_f32; cfg.n_features];
        let bn_var = vec![1.0_f32; cfg.n_features];

        // Prior scales P_i (start at all-ones)
        let mut prior = vec![1.0_f32; cfg.n_features];

        // Accumulated step outputs for final aggregation
        let mut step_outputs: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_steps);
        let mut attention_masks = vec![0.0_f32; cfg.n_steps * cfg.n_features];

        // Initial h (step input) — zeros
        let mut h = vec![0.0_f32; na_nd];

        for step in 0..cfg.n_steps {
            // Attention transform: att_logits = W_att * h + b_att
            let att_logits = Self::matvec(
                &self.att_w[step],
                &self.att_b[step],
                &h,
                na_nd,
                cfg.n_features,
            );

            // Apply prior scale (element-wise multiply) then BN
            let scaled: Vec<f32> = prior
                .iter()
                .zip(att_logits.iter())
                .map(|(&p, &a)| p * a)
                .collect();
            let bn_out = self.bn.forward_inference(&scaled, &bn_mean, &bn_var)?;

            // Sparsemax → attention mask M_i
            let mask = sparsemax(&bn_out)?;

            // Update prior: P_{i+1} = P_i ⊙ (γ - M_i)
            for f in 0..cfg.n_features {
                prior[f] *= cfg.gamma - mask[f];
            }

            // Store attention mask
            attention_masks[step * cfg.n_features..(step + 1) * cfg.n_features]
                .copy_from_slice(&mask);

            // Feature selection: h_sel = M_i ⊙ x
            let h_sel: Vec<f32> = mask.iter().zip(x.iter()).map(|(&m, &xi)| m * xi).collect();

            // Shared transform: [n_features] → [2*(n_d + n_a)] → GLU → [n_d + n_a]
            let shared_out = Self::matvec(
                &self.shared_w,
                &self.shared_b,
                &h_sel,
                cfg.n_features,
                2 * na_nd,
            );
            let shared_glu = glu(&shared_out)?;

            // Step-specific transform: [n_d + n_a] → [2*(n_d + n_a)] → GLU → [n_d + n_a]
            let step_out = Self::matvec(
                &self.step_w[step],
                &self.step_b[step],
                &shared_glu,
                na_nd,
                2 * na_nd,
            );
            let step_glu = glu(&step_out)?;

            // ReLU and accumulate
            let relu_out: Vec<f32> = step_glu.iter().map(|&v| v.max(0.0)).collect();
            h = relu_out;

            // Keep only the n_d portion for output aggregation
            step_outputs.push(h[..cfg.n_d].to_vec());
        }

        // Aggregate: mean over steps
        let mut agg = vec![0.0_f32; cfg.n_d];
        for so in &step_outputs {
            for (a, &v) in agg.iter_mut().zip(so.iter()) {
                *a += v;
            }
        }
        let n_steps_f = cfg.n_steps as f32;
        for a in &mut agg {
            *a /= n_steps_f;
        }

        // Output head
        let logits = Self::matvec(&self.final_w, &self.final_b, &agg, cfg.n_d, cfg.n_classes);

        Ok((logits, attention_masks))
    }

    /// Batch forward: `x` is flat `[batch_size * n_features]`.
    ///
    /// Returns `(logits [batch * n_classes], attention_masks [batch * n_steps * n_features])`.
    pub fn forward_batch(
        &self,
        x: &[f32],
        batch_size: usize,
    ) -> TabularResult<(Vec<f32>, Vec<f32>)> {
        let cfg = &self.config;
        let n_feat = cfg.n_features;
        if x.len() != batch_size * n_feat {
            return Err(TabularError::DimensionMismatch {
                expected: batch_size * n_feat,
                got: x.len(),
            });
        }
        let mut all_logits = Vec::with_capacity(batch_size * cfg.n_classes);
        let mut all_masks = Vec::with_capacity(batch_size * cfg.n_steps * n_feat);

        for b in 0..batch_size {
            let row = &x[b * n_feat..(b + 1) * n_feat];
            let (logits, masks) = self.forward(row)?;
            all_logits.extend_from_slice(&logits);
            all_masks.extend_from_slice(&masks);
        }
        Ok((all_logits, all_masks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn glu_halves_dim() {
        let x = vec![1.0_f32; 8];
        let out = glu(&x).expect("glu should succeed");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn glu_odd_error() {
        let x = vec![1.0_f32; 7];
        assert!(glu(&x).is_err());
    }

    #[test]
    fn tabnet_forward_shape() {
        let mut rng = LcgRng::new(42);
        let cfg = TabNetConfig {
            n_features: 8,
            n_d: 4,
            n_a: 4,
            n_steps: 3,
            gamma: 1.5,
            n_classes: 2,
        };
        let layer = TabNetLayer::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.5_f32; 8];
        let (logits, masks) = layer.forward(&x).expect("forward should succeed");
        assert_eq!(logits.len(), 2);
        assert_eq!(masks.len(), 3 * 8);
    }

    #[test]
    fn tabnet_attention_non_negative() {
        let mut rng = LcgRng::new(99);
        let cfg = TabNetConfig {
            n_features: 6,
            n_d: 4,
            n_a: 4,
            n_steps: 3,
            gamma: 1.5,
            n_classes: 2,
        };
        let layer = TabNetLayer::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.1_f32; 6];
        let (_, masks) = layer.forward(&x).expect("forward should succeed");
        assert!(masks.iter().all(|&v| v >= 0.0));
    }
}
