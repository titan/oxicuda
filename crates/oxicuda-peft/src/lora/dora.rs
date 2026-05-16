use crate::handle::LcgRng;
use crate::lora::lora::{LoraConfig, mat_vec_mul};

/// DoRA (Weight-Decomposed Low-Rank Adaptation) linear layer.
///
/// Decomposes the pre-trained weight into column-wise magnitude `m` and direction `V = W / m`,
/// then adapts the direction with a LoRA term: `adapted = V + scale·B·A`,
/// then re-normalises per-column and rescales by the learned magnitude vector.
///
/// W shape: `[out_features × in_features]` (row-major, columns = input features).
/// `magnitude`: shape `[in_features]` — one scalar per input column of the weight.
/// `direction_w`: shape `[out_features × in_features]` — normalised weight directions.
/// `a`: shape `[rank × in_features]`.
/// `b`: shape `[out_features × rank]`.
#[derive(Debug, Clone)]
pub struct DoraLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// LoRA rank.
    pub rank: usize,
    /// Effective LoRA scale α/r.
    pub scale: f32,
    /// Per-column magnitude vector, shape `[in_features]`.
    pub magnitude: Vec<f32>,
    /// Column-normalised weight direction, shape `[out_features × in_features]`.
    pub direction_w: Vec<f32>,
    /// LoRA factor A, shape `[rank × in_features]`.
    pub a: Vec<f32>,
    /// LoRA factor B, shape `[out_features × rank]`.
    pub b: Vec<f32>,
}

impl DoraLinear {
    /// Construct a `DoraLinear` from a pre-trained weight matrix.
    ///
    /// `w` must have length `out_features * in_features` (row-major `[out × in]`).
    #[must_use]
    pub fn from_pretrained(
        w: &[f32],
        in_features: usize,
        out_features: usize,
        cfg: &LoraConfig,
        rng: &mut LcgRng,
    ) -> Self {
        let scale = cfg.alpha / cfg.r as f32;

        // Compute per-column L2 norms (columns of W correspond to input features).
        // W[i, j] = w[i * in_features + j]; column j spans rows 0..out_features.
        let mut magnitude = vec![0.0_f32; in_features];
        for j in 0..in_features {
            let norm_sq: f32 = (0..out_features)
                .map(|i| w[i * in_features + j].powi(2))
                .sum();
            magnitude[j] = norm_sq.sqrt().max(1e-12);
        }

        // Compute direction: V[i, j] = W[i, j] / magnitude[j]
        let mut direction_w = vec![0.0_f32; out_features * in_features];
        for i in 0..out_features {
            for j in 0..in_features {
                direction_w[i * in_features + j] = w[i * in_features + j] / magnitude[j];
            }
        }

        // Initialise LoRA factors
        let mut a = vec![0.0_f32; cfg.r * in_features];
        rng.fill_normal(&mut a);
        for v in a.iter_mut() {
            *v *= cfg.init_scale;
        }
        let b = vec![0.0_f32; out_features * cfg.r];

        Self {
            in_features,
            out_features,
            rank: cfg.r,
            scale,
            magnitude,
            direction_w,
            a,
            b,
        }
    }

    /// Compute the DoRA forward pass.
    ///
    /// Computes `adapted = direction_w + scale·B·A`, then re-normalises each column
    /// by `magnitude / col_norm`, then multiplies by `x`.
    ///
    /// `x` must have length `in_features`. Returns a vector of length `out_features`.
    #[must_use]
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Compute LoRA delta matrix: scale · B · A (shape [out × in])
        let lora_delta = self.compute_lora_delta();

        // Adapted weight: V + scale·B·A
        let mut adapted: Vec<f32> = self
            .direction_w
            .iter()
            .zip(lora_delta.iter())
            .map(|(v, d)| v + d)
            .collect();

        // Re-normalise adapted columns and rescale by magnitude
        for j in 0..self.in_features {
            // Compute L2 norm of column j in adapted
            let col_norm_sq: f32 = (0..self.out_features)
                .map(|i| adapted[i * self.in_features + j].powi(2))
                .sum();
            let col_norm = col_norm_sq.sqrt().max(1e-12);
            let rescale = self.magnitude[j] / col_norm;
            for i in 0..self.out_features {
                adapted[i * self.in_features + j] *= rescale;
            }
        }

        mat_vec_mul(&adapted, x, self.out_features, self.in_features)
    }

    /// Compute `scale · B · A` as a flat `[out_features × in_features]` matrix.
    fn compute_lora_delta(&self) -> Vec<f32> {
        let mut delta = vec![0.0_f32; self.out_features * self.in_features];
        for i in 0..self.out_features {
            for k in 0..self.rank {
                let b_ik = self.b[i * self.rank + k];
                if b_ik == 0.0 {
                    continue;
                }
                for j in 0..self.in_features {
                    delta[i * self.in_features + j] +=
                        self.scale * b_ik * self.a[k * self.in_features + j];
                }
            }
        }
        delta
    }
}
