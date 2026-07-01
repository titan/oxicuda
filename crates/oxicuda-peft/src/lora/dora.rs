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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::lora::lora::LoraConfig;

    fn lora_cfg(r: usize, alpha: f32) -> LoraConfig {
        LoraConfig {
            r,
            alpha,
            init_scale: 0.01,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: direction_w has unit L2 norm per column after construction
    // -----------------------------------------------------------------------
    #[test]
    fn direction_w_has_unit_column_norms() {
        // W = [1,2,3,4,5,6] viewed as [2 rows × 3 cols]
        let w: Vec<f32> = (1..=6).map(|v| v as f32).collect();
        let mut rng = LcgRng::new(1);
        let layer = DoraLinear::from_pretrained(&w, 3, 2, &lora_cfg(1, 1.0), &mut rng);
        for j in 0..3usize {
            let norm_sq: f32 = (0..2usize)
                .map(|i| layer.direction_w[i * 3 + j].powi(2))
                .sum();
            let norm = norm_sq.sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "direction_w column {j}: expected unit norm, got {norm}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: magnitude equals the per-column L2 norm of the input weight W
    // -----------------------------------------------------------------------
    #[test]
    fn magnitude_equals_column_norm_of_w() {
        // W = [[3,4],[0,0]]; column norms: col0=3.0, col1=4.0
        let w = vec![3.0_f32, 4.0, 0.0, 0.0];
        let mut rng = LcgRng::new(2);
        let layer = DoraLinear::from_pretrained(&w, 2, 2, &lora_cfg(1, 1.0), &mut rng);
        assert!(
            (layer.magnitude[0] - 3.0).abs() < 1e-5,
            "magnitude[0] expected 3.0, got {}",
            layer.magnitude[0]
        );
        assert!(
            (layer.magnitude[1] - 4.0).abs() < 1e-5,
            "magnitude[1] expected 4.0, got {}",
            layer.magnitude[1]
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: with B=0 (zero-initialised by construction), forward = W · x
    // -----------------------------------------------------------------------
    #[test]
    fn zero_lora_recovers_base_weight() {
        // W = [[1,2],[3,4]], out=2, in=2
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut rng = LcgRng::new(3);
        // B is zero-initialised; magnitude × normalised-direction = W exactly
        let layer = DoraLinear::from_pretrained(&w, 2, 2, &lora_cfg(1, 1.0), &mut rng);
        let x = vec![1.0_f32, 2.0];
        let y = layer.forward(&x);
        // W·x = [1*1+2*2, 3*1+4*2] = [5, 11]
        assert_eq!(y.len(), 2);
        assert!((y[0] - 5.0).abs() < 1e-4, "y[0] expected 5.0, got {}", y[0]);
        assert!(
            (y[1] - 11.0).abs() < 1e-4,
            "y[1] expected 11.0, got {}",
            y[1]
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: output vector has length out_features
    // -----------------------------------------------------------------------
    #[test]
    fn forward_output_shape() {
        let w = vec![0.5_f32; 5 * 4];
        let mut rng = LcgRng::new(4);
        let layer = DoraLinear::from_pretrained(&w, 4, 5, &lora_cfg(2, 4.0), &mut rng);
        assert_eq!(layer.forward(&[1.0_f32; 4]).len(), 5);
    }

    // -----------------------------------------------------------------------
    // Test 5: same seed + same input → identical output (determinism)
    // -----------------------------------------------------------------------
    #[test]
    fn forward_deterministic() {
        let w: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let mut rng1 = LcgRng::new(77);
        let mut rng2 = LcgRng::new(77);
        let l1 = DoraLinear::from_pretrained(&w, 4, 3, &lora_cfg(2, 4.0), &mut rng1);
        let l2 = DoraLinear::from_pretrained(&w, 4, 3, &lora_cfg(2, 4.0), &mut rng2);
        let x = vec![0.1_f32, -0.3, 0.5, 0.7];
        assert_eq!(l1.forward(&x), l2.forward(&x));
    }

    // -----------------------------------------------------------------------
    // Test 6: forward matches explicit DoRA decomposition for a tiny case
    //
    // W = I₂ (identity), rank=1, alpha=2 (scale=2), B=[[0],[1]], A=[[1,0]].
    // adapted = direction_W + scale·B·A = I + 2·[[0,0],[1,0]] = [[1,0],[2,1]]
    // Column norms of adapted: col0=√5, col1=1. magnitude=[1,1] (identity W).
    // Effective W = [[1/√5,0],[2/√5,1]], so y = [1/√5, 2/√5+1] for x=[1,1].
    // -----------------------------------------------------------------------
    #[test]
    fn forward_matches_explicit_decomposition() {
        let w = vec![1.0_f32, 0.0, 0.0, 1.0]; // identity, out=2, in=2
        let mut rng = LcgRng::new(5);
        let cfg = LoraConfig {
            r: 1,
            alpha: 2.0,
            init_scale: 0.0,
        };
        let mut layer = DoraLinear::from_pretrained(&w, 2, 2, &cfg, &mut rng);
        // B shape [out×rank]=[2×1]: B[0,0]=0, B[1,0]=1
        layer.b = vec![0.0, 1.0];
        // A shape [rank×in]=[1×2]: A[0,0]=1, A[0,1]=0
        layer.a = vec![1.0, 0.0];

        let x = vec![1.0_f32, 1.0];
        let y = layer.forward(&x);

        let sqrt5 = 5.0_f32.sqrt();
        let expected_y0 = 1.0_f32 / sqrt5;
        let expected_y1 = 2.0_f32 / sqrt5 + 1.0;

        assert_eq!(y.len(), 2);
        assert!(
            (y[0] - expected_y0).abs() < 1e-5,
            "y[0] expected {expected_y0}, got {}",
            y[0]
        );
        assert!(
            (y[1] - expected_y1).abs() < 1e-5,
            "y[1] expected {expected_y1}, got {}",
            y[1]
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: all output elements are finite for random initialisation
    // -----------------------------------------------------------------------
    #[test]
    fn forward_finite_outputs() {
        let w: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.3).collect();
        let mut rng = LcgRng::new(88);
        let layer = DoraLinear::from_pretrained(&w, 5, 4, &lora_cfg(3, 6.0), &mut rng);
        let x = vec![0.5_f32, -0.3, 1.0, 0.2, -0.8];
        for &v in layer.forward(&x).iter() {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }
}
