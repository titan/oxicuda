//! DoRA (Weight-Decomposed Low-Rank Adaptation) with row-wise magnitude decomposition.
//!
//! Unlike column-wise DoRA (which decomposes per input feature), this variant decomposes
//! per output feature (row-wise), giving a magnitude vector of shape `[out_features]`.
//! This matches the original DoRA paper's decomposition of W = m * (W / ||W||_row).

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Configuration for a `DoraLayer`.
#[derive(Debug, Clone)]
pub struct DoraConfig {
    /// Number of input features (columns of the weight matrix).
    pub in_features: usize,
    /// Number of output features (rows of the weight matrix).
    pub out_features: usize,
    /// LoRA rank r (must be > 0 and ≤ min(in, out)).
    pub rank: usize,
    /// LoRA alpha scaling factor; effective scale = alpha / rank.
    pub alpha: f32,
}

/// DoRA linear layer with row-wise magnitude decomposition.
///
/// Decomposes the weight matrix W₀ ∈ ℝ^{out × in} as:
///   W₀ = m ⊙ (W₀ / row_norms(W₀))
///
/// where `m ∈ ℝ^{out}` is a learnable magnitude vector initialized to the row norms of W₀,
/// and B ∈ ℝ^{out × rank}, A ∈ ℝ^{rank × in} are learnable low-rank factors.
///
/// Forward pass:
/// 1. delta_w = (alpha/rank) · B @ A          — [out × in]
/// 2. adapted = W₀ + delta_w                  — [out × in]
/// 3. row_norms = `||adapted[i,:]||₂`            — `[out]`
/// 4. normalized = `adapted[i,:] / row_norms[i]` — `[out × in]`
/// 5. scaled = `m[i] · normalized[i,:]`          — `[out × in]`
/// 6. output = scaled @ x                      — [out × batch]
#[derive(Debug, Clone)]
pub struct DoraLayer {
    /// Magnitude per output row, shape `[out_features]`. Learnable.
    pub m: Vec<f32>,
    /// LoRA factor B, shape `[out_features × rank]` (row-major). Learnable, zero-initialized.
    pub b: Vec<f32>,
    /// LoRA factor A, shape `[rank × in_features]` (row-major). Learnable.
    pub a: Vec<f32>,
    /// Frozen base weight W₀, shape `[out_features × in_features]` (row-major).
    pub w0: Vec<f32>,
    /// Layer configuration.
    pub config: DoraConfig,
}

impl DoraLayer {
    /// Construct a `DoraLayer` with random W₀ and properly initialized magnitude.
    ///
    /// Initialization:
    /// - `w0` ~ N(0, 0.01²)
    /// - `m[i]` = `||w0[i,:]||₂`, clamped to a minimum of 1e-12
    /// - `a` ~ N(0, 0.02²)
    /// - `b` = 0 (zero-initialized so delta_w = 0 at step 0)
    ///
    /// # Errors
    /// - `PeftError::Internal` if `rank == 0`, `in_features == 0`, or `out_features == 0`.
    /// - `PeftError::RankTooLarge` if `rank > min(in_features, out_features)`.
    pub fn new(config: DoraConfig, rng: &mut LcgRng) -> PeftResult<Self> {
        if config.rank == 0 {
            return Err(PeftError::Internal {
                msg: "rank must be > 0".into(),
            });
        }
        if config.in_features == 0 {
            return Err(PeftError::Internal {
                msg: "in_features must be > 0".into(),
            });
        }
        if config.out_features == 0 {
            return Err(PeftError::Internal {
                msg: "out_features must be > 0".into(),
            });
        }
        let min_dim = config.in_features.min(config.out_features);
        if config.rank > min_dim {
            return Err(PeftError::RankTooLarge {
                rank: config.rank,
                dim: min_dim,
            });
        }

        // Initialize w0 ~ N(0, 0.01^2)
        let w0_size = config.out_features * config.in_features;
        let mut w0 = vec![0.0_f32; w0_size];
        rng.fill_normal(&mut w0);
        for v in w0.iter_mut() {
            *v *= 0.01;
        }

        // Initialize m = ||w0[i,:]||_2 for each row i
        let m = compute_row_norms(&w0, config.out_features, config.in_features);

        // Initialize a ~ N(0, 0.02^2)
        let a_size = config.rank * config.in_features;
        let mut a = vec![0.0_f32; a_size];
        rng.fill_normal(&mut a);
        for v in a.iter_mut() {
            *v *= 0.02;
        }

        // Initialize b = 0
        let b = vec![0.0_f32; config.out_features * config.rank];

        Ok(Self {
            m,
            b,
            a,
            w0,
            config,
        })
    }

    /// Perform the DoRA forward pass.
    ///
    /// `x` has shape `[in_features × batch_size]` (column-major: each column is one sample).
    /// Returns a `Vec<f32>` of shape `[out_features × batch_size]`.
    ///
    /// # Errors
    /// - `PeftError::DimensionMismatch` if `x.len() != in_features * batch_size`.
    /// - `PeftError::EmptyInput` if `x` is empty.
    pub fn forward(&self, x: &[f32], batch_size: usize) -> PeftResult<Vec<f32>> {
        let in_f = self.config.in_features;
        let out_f = self.config.out_features;
        let rank = self.config.rank;

        if x.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        let expected = in_f * batch_size;
        if x.len() != expected {
            return Err(PeftError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let scale = self.config.alpha / rank as f32;

        // Step 1: delta_w[out × in] = scale * B[out × rank] @ A[rank × in]
        // B[i, k] = b[i * rank + k], A[k, j] = a[k * in_f + j]
        let mut delta_w = vec![0.0_f32; out_f * in_f];
        for i in 0..out_f {
            for j in 0..in_f {
                let mut acc = 0.0_f32;
                for k in 0..rank {
                    acc += self.b[i * rank + k] * self.a[k * in_f + j];
                }
                delta_w[i * in_f + j] = scale * acc;
            }
        }

        // Step 2: adapted[out × in] = w0 + delta_w
        let mut adapted = self.w0.clone();
        for (v, d) in adapted.iter_mut().zip(delta_w.iter()) {
            *v += d;
        }

        // Step 3: row_norms[out] = ||adapted[i,:]||_2
        let row_norms = compute_row_norms(&adapted, out_f, in_f);

        // Step 4: normalized[i, j] = adapted[i, j] / max(row_norms[i], 1e-12)
        // Step 5: scaled[i, j] = m[i] * normalized[i, j]
        // Combine into one pass: scaled[i, j] = m[i] * adapted[i, j] / row_norms[i]
        let mut scaled = vec![0.0_f32; out_f * in_f];
        for i in 0..out_f {
            let denom = row_norms[i].max(1e-12_f32);
            let mi = self.m[i];
            for j in 0..in_f {
                scaled[i * in_f + j] = mi * adapted[i * in_f + j] / denom;
            }
        }

        // Step 6: output[out × batch] = scaled[out × in] @ x[in × batch]
        // output[i, b] = sum_j scaled[i, j] * x[j * batch_size + b]
        // Note: x is stored as [in × batch] in column-major order: x[j, b] = x[j * batch_size + b]
        let mut output = vec![0.0_f32; out_f * batch_size];
        for i in 0..out_f {
            for b_idx in 0..batch_size {
                let mut acc = 0.0_f32;
                for j in 0..in_f {
                    acc += scaled[i * in_f + j] * x[j * batch_size + b_idx];
                }
                output[i * batch_size + b_idx] = acc;
            }
        }

        Ok(output)
    }

    /// Total number of trainable parameters.
    ///
    /// = `rank * in_features` (A) + `out_features * rank` (B) + `out_features` (m)
    #[must_use]
    pub fn n_trainable_params(&self) -> usize {
        self.config.rank * self.config.in_features
            + self.config.out_features * self.config.rank
            + self.config.out_features
    }

    /// Borrow the magnitude vector `m`.
    #[must_use]
    pub fn magnitude(&self) -> &[f32] {
        &self.m
    }

    /// Borrow the frozen base weight W₀.
    #[must_use]
    pub fn w0_ref(&self) -> &[f32] {
        &self.w0
    }
}

/// Compute L2 row norms for a matrix stored in row-major order.
/// Returns `[n_rows]` where each entry is `max(||row_i||_2, 1e-12)`.
fn compute_row_norms(mat: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let mut norms = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let row_start = i * n_cols;
        let row_end = row_start + n_cols;
        let sq_sum: f32 = mat[row_start..row_end].iter().map(|&v| v * v).sum();
        norms.push(sq_sum.sqrt().max(1e-12_f32));
    }
    norms
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(in_f: usize, out_f: usize, rank: usize, alpha: f32, seed: u64) -> DoraLayer {
        let mut rng = LcgRng::new(seed);
        let cfg = DoraConfig {
            in_features: in_f,
            out_features: out_f,
            rank,
            alpha,
        };
        DoraLayer::new(cfg, &mut rng).expect("valid DoraLayer config")
    }

    /// Build column-major [in × batch] input of ones.
    fn ones_input(in_f: usize, batch: usize) -> Vec<f32> {
        vec![1.0_f32; in_f * batch]
    }

    #[test]
    fn output_shape() {
        let layer = make_layer(16, 8, 4, 8.0, 1);
        let x = ones_input(16, 3);
        let out = layer.forward(&x, 3).expect("forward ok");
        assert_eq!(
            out.len(),
            8 * 3,
            "expected output [out_features × batch_size] = 24, got {}",
            out.len()
        );
    }

    #[test]
    fn output_finite() {
        let layer = make_layer(12, 6, 3, 6.0, 2);
        let x: Vec<f32> = (0..12 * 2).map(|i| i as f32 * 0.05 - 0.3).collect();
        let out = layer.forward(&x, 2).expect("forward ok");
        for (idx, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{idx}] is not finite: {v}");
        }
    }

    #[test]
    fn n_params_correct() {
        let in_f = 16usize;
        let out_f = 8usize;
        let rank = 4usize;
        let layer = make_layer(in_f, out_f, rank, 8.0, 3);
        let expected = rank * in_f + out_f * rank + out_f;
        assert_eq!(
            layer.n_trainable_params(),
            expected,
            "n_trainable_params mismatch"
        );
    }

    #[test]
    fn rank_0_error() {
        let mut rng = LcgRng::new(4);
        let cfg = DoraConfig {
            in_features: 8,
            out_features: 8,
            rank: 0,
            alpha: 4.0,
        };
        let result = DoraLayer::new(cfg, &mut rng);
        assert!(result.is_err(), "rank=0 should return an error");
    }

    #[test]
    fn alpha_0_zeros_delta_w() {
        // With alpha=0, scale=0, so delta_w=0 and adapted=w0
        // The output equals m ⊙ (w0 / row_norms(w0)) @ x = normalized_w0 @ x
        // (where normalized_w0 has unit-norm rows scaled by m)
        let layer = make_layer(8, 4, 2, 0.0, 5);
        let x = ones_input(8, 1);
        let out = layer.forward(&x, 1).expect("forward ok");
        assert_eq!(out.len(), 4);
        // Verify that with alpha=0, the output is computed from w0 directly
        // (no LoRA contribution), all values should be finite
        for &v in &out {
            assert!(v.is_finite(), "alpha=0 output not finite: {v}");
        }
        // Also verify: row norms of adapted (= w0) * scaling match magnitude
        // output[i] = m[i] * (sum_j w0[i,j]/row_norm(w0[i,:])) where x=ones
        let in_f = 8usize;
        let out_f = 4usize;
        let row_norms = compute_row_norms(&layer.w0, out_f, in_f);
        for i in 0..out_f {
            let row_sum: f32 = layer.w0[i * in_f..(i + 1) * in_f].iter().sum();
            let expected_i = layer.magnitude()[i] * row_sum / row_norms[i];
            assert!(
                (out[i] - expected_i).abs() < 1e-4,
                "alpha=0: output[{i}]={} expected {expected_i}",
                out[i]
            );
        }
    }

    #[test]
    fn col_norms_are_1() {
        // After normalization (step 4), each row of `normalized` should have unit norm.
        // We verify this indirectly: if m[i] = row_norm(w0[i,:]) and alpha=0,
        // then scaled[i,:] = m[i] * normalized[i,:] and ||normalized[i,:]||_2 = 1.
        let in_f = 10usize;
        let out_f = 5usize;
        let mut rng = LcgRng::new(6);
        let cfg = DoraConfig {
            in_features: in_f,
            out_features: out_f,
            rank: 2,
            alpha: 0.0,
        };
        let layer = DoraLayer::new(cfg, &mut rng).expect("valid config");
        // With alpha=0: adapted = w0; normalized[i,:] = w0[i,:] / ||w0[i,:]||
        let row_norms = compute_row_norms(&layer.w0, out_f, in_f);
        for (i, &rn) in row_norms.iter().enumerate() {
            let row_sq_sum: f32 = (0..in_f)
                .map(|j| {
                    let v = layer.w0[i * in_f + j] / rn;
                    v * v
                })
                .sum();
            let norm = row_sq_sum.sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "normalized row {i} has norm {norm}, expected 1.0"
            );
        }
    }

    #[test]
    fn m_initialized_to_w0_norms() {
        // Verify m[i] = ||w0[i,:]||_2 for all rows i
        let in_f = 8usize;
        let out_f = 4usize;
        let mut rng = LcgRng::new(7);
        let cfg = DoraConfig {
            in_features: in_f,
            out_features: out_f,
            rank: 2,
            alpha: 4.0,
        };
        let layer = DoraLayer::new(cfg, &mut rng).expect("valid config");
        let expected_norms = compute_row_norms(&layer.w0, out_f, in_f);
        let m = layer.magnitude();
        for i in 0..out_f {
            assert!(
                (m[i] - expected_norms[i]).abs() < 1e-6,
                "m[{i}]={} but ||w0[{i},:]||_2={}",
                m[i],
                expected_norms[i]
            );
        }
    }

    #[test]
    fn dora_differs_from_lora() {
        // DoRA applies row-wise normalization + magnitude rescaling on adapted = w0 + delta_w.
        // With non-zero B (so delta_w ≠ 0), the adapted matrix differs from w0,
        // which means adapted / row_norm(adapted) ≠ w0 / row_norm(w0).
        // Hence the DoRA output m ⊙ normalized(adapted) @ x differs from
        // the plain LoRA output (w0 + delta_w) @ x.
        let in_f = 8usize;
        let out_f = 4usize;
        let rank = 2usize;
        let alpha = 4.0_f32;
        let mut rng = LcgRng::new(8);
        let cfg = DoraConfig {
            in_features: in_f,
            out_features: out_f,
            rank,
            alpha,
        };
        let mut layer = DoraLayer::new(cfg, &mut rng).expect("valid config");
        // Give B non-trivial values so the LoRA delta_w is non-zero
        for (k, v) in layer.b.iter_mut().enumerate() {
            *v = (k as f32 + 1.0) * 0.1;
        }
        let x: Vec<f32> = (0..in_f).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let dora_out = layer.forward(&x, 1).expect("dora forward ok");

        // Compute plain LoRA output: (w0 + scale * B @ A) @ x (without row normalization)
        let scale = alpha / rank as f32;
        let mut plain_out = vec![0.0_f32; out_f];
        for (i, po) in plain_out.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for (j, &xj) in x.iter().enumerate() {
                let mut w_ij = layer.w0[i * in_f + j];
                for k in 0..rank {
                    w_ij += scale * layer.b[i * rank + k] * layer.a[k * in_f + j];
                }
                acc += w_ij * xj;
            }
            *po = acc;
        }

        let diff: f32 = dora_out
            .iter()
            .zip(plain_out.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "DoRA and plain LoRA outputs should differ (normalization + magnitude rescaling), diff={diff}"
        );
    }

    #[test]
    fn batch_size_varies() {
        // batch_size=1 and batch_size=4 both work and produce correct output lengths
        let in_f = 6usize;
        let out_f = 3usize;
        let mut rng = LcgRng::new(9);
        let cfg = DoraConfig {
            in_features: in_f,
            out_features: out_f,
            rank: 2,
            alpha: 4.0,
        };
        let layer = DoraLayer::new(cfg, &mut rng).expect("valid config");

        let x1 = ones_input(in_f, 1);
        let out1 = layer.forward(&x1, 1).expect("batch=1 ok");
        assert_eq!(
            out1.len(),
            out_f,
            "batch=1: output length should be {out_f}"
        );

        let x4 = ones_input(in_f, 4);
        let out4 = layer.forward(&x4, 4).expect("batch=4 ok");
        assert_eq!(
            out4.len(),
            out_f * 4,
            "batch=4: output length should be {}",
            out_f * 4
        );

        // All outputs should be finite
        for &v in out1.iter().chain(out4.iter()) {
            assert!(v.is_finite(), "batch varies: output not finite: {v}");
        }
    }

    #[test]
    fn dimension_mismatch_error() {
        let layer = make_layer(8, 4, 2, 4.0, 10);
        // Provide wrong x length
        let x = vec![0.0_f32; 5]; // should be 8 * batch_size
        let result = layer.forward(&x, 1);
        assert!(
            result.is_err(),
            "wrong x length should return DimensionMismatch error"
        );
    }
}
