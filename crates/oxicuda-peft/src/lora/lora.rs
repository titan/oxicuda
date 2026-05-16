use crate::handle::LcgRng;

/// Configuration for a LoRA adapter.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Intrinsic rank of the low-rank decomposition.
    pub r: usize,
    /// LoRA scaling factor α; the effective scale is α/r.
    pub alpha: f32,
    /// Standard deviation used to initialise matrix A.
    pub init_scale: f32,
}

/// Low-rank adaptation of a single linear layer: ΔW = scale · B · A.
///
/// W is the base weight (frozen), A and B are the trainable low-rank factors.
/// W shape: `[out_features × in_features]` (row-major).
/// A shape: `[rank × in_features]`.
/// B shape: `[out_features × rank]`.
#[derive(Debug, Clone)]
pub struct LoraLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Rank of the decomposition.
    pub rank: usize,
    /// Effective scale = α / r.
    pub scale: f32,
    /// Base weight matrix, shape `[out_features × in_features]`.
    pub w: Vec<f32>,
    /// Low-rank factor A, shape `[rank × in_features]`.
    pub a: Vec<f32>,
    /// Low-rank factor B, shape `[out_features × rank]`.
    pub b: Vec<f32>,
}

impl LoraLinear {
    /// Construct a new `LoraLinear`.
    ///
    /// W is zero-initialised. A is sampled from N(0, `cfg.init_scale`). B is zero-initialised.
    #[must_use]
    pub fn new(
        in_features: usize,
        out_features: usize,
        cfg: &LoraConfig,
        rng: &mut LcgRng,
    ) -> Self {
        let scale = cfg.alpha / cfg.r as f32;
        let w = vec![0.0_f32; out_features * in_features];
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
            w,
            a,
            b,
        }
    }

    /// Compute the forward pass: `(W + scale·B·A) · x`.
    ///
    /// `x` must have length `in_features`. Returns a vector of length `out_features`.
    #[must_use]
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Base: out = W · x
        let mut out = mat_vec_mul(&self.w, x, self.out_features, self.in_features);
        // LoRA delta: tmp = A · x  (shape [rank])
        let tmp = mat_vec_mul(&self.a, x, self.rank, self.in_features);
        // delta = B · tmp  (shape [out_features])
        let delta = mat_vec_mul(&self.b, &tmp, self.out_features, self.rank);
        for (o, d) in out.iter_mut().zip(delta.iter()) {
            *o += self.scale * d;
        }
        out
    }

    /// Merge the LoRA adapter into the base weight: `W += scale · B · A`.
    pub fn merge_into_w(&mut self) {
        let delta = self.lora_delta();
        for (w, d) in self.w.iter_mut().zip(delta.iter()) {
            *w += d;
        }
    }

    /// Subtract the previously merged adapter from the base weight: `W -= scale · B · A`.
    pub fn unmerge_from_w(&mut self) {
        let delta = self.lora_delta();
        for (w, d) in self.w.iter_mut().zip(delta.iter()) {
            *w -= d;
        }
    }

    /// Compute the full LoRA delta matrix `scale · B · A` as a flat `[out_features × in_features]` matrix.
    #[must_use]
    pub fn lora_delta(&self) -> Vec<f32> {
        // result[i, j] = scale * sum_r B[i, r] * A[r, j]
        let mut result = vec![0.0_f32; self.out_features * self.in_features];
        for i in 0..self.out_features {
            for k in 0..self.rank {
                let b_ik = self.b[i * self.rank + k];
                for j in 0..self.in_features {
                    result[i * self.in_features + j] +=
                        self.scale * b_ik * self.a[k * self.in_features + j];
                }
            }
        }
        result
    }
}

/// Multiply matrix `m` (shape `[rows × cols]`, row-major) by column vector `v` (length `cols`).
pub(crate) fn mat_vec_mul(m: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|i| {
            let row_start = i * cols;
            m[row_start..row_start + cols]
                .iter()
                .zip(v.iter())
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}
