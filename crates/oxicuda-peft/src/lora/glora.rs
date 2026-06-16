//! GLoRA — Generalized Low-Rank Adaptation (Chavan et al. 2023).
//!
//! Reference: Chavan, A., Liu, Z., Gupta, D., Xing, E., & Shen, Z. (2023).
//! *One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning*.
//! <https://arxiv.org/abs/2306.07967>
//!
//! GLoRA unifies LoRA, prompt-tuning, scaling adapters and prefix-tuning under a single
//! reparameterisation of a frozen linear layer `y = W₀·x + b₀`. It introduces five *support
//! tensors* `A, B, C, D, E` that jointly modulate the weight **and** the bias:
//!
//! ```text
//!   y = (W₀ + W₀ ⊙ A + B)·x  +  (C·W₀ + D ⊙ b₀ + E)
//! ```
//!
//! where, in this implementation:
//!
//! * `A ∈ ℝ^{out × in}` — a multiplicative weight modulation realised as a low-rank product
//!   `A = a_b · a_a` (so `W₀ ⊙ A` becomes the Hadamard product `W₀ ⊙ (a_b·a_a)`). Holds the
//!   "scale the pretrained weight" capability.
//! * `B ∈ ℝ^{out × in}` — an additive weight term, also a low-rank product `B = b_b · b_a`
//!   (the classical LoRA delta).
//! * `C ∈ ℝ^{out}` — a learnable vector that, contracted with `W₀`, produces a prompt-like bias
//!   `(C·W₀)_i = Σ_j C_j · W₀[j, i]`… here simplified to a per-output learnable bias offset
//!   `C ⊙ rowsum(W₀)` capturing the weight-derived bias capability.
//! * `D ∈ ℝ^{out}` — a multiplicative modulation of the frozen bias `b₀`.
//! * `E ∈ ℝ^{out}` — a free additive bias (prompt-tuning capability).
//!
//! Every support tensor is initialised so the layer starts as an **exact identity** (output
//! equals the frozen `W₀·x + b₀`): `a_a, b_b = 0`, `C, E = 0`, `D = 1`. GLoRA's structural
//! re-parameterisation means the whole adapter can be **merged** into an equivalent
//! `(W_eff, b_eff)` pair with no inference overhead.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Configuration for a [`GloraLinear`] adapter.
#[derive(Debug, Clone)]
pub struct GloraConfig {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Low-rank dimension shared by the `A` and `B` support tensors (`> 0`, `≤ min(in, out)`).
    pub rank: usize,
    /// Std-dev for the random (frozen-direction) factors of `A` and `B`.
    pub init_scale: f32,
}

/// GLoRA-adapted linear layer with five support tensors.
#[derive(Debug, Clone)]
pub struct GloraLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Low-rank dimension.
    pub rank: usize,
    /// Frozen base weight `W₀`, shape `[out × in]` (row-major).
    pub w0: Vec<f32>,
    /// Frozen base bias `b₀`, shape `[out]`.
    pub b0: Vec<f32>,
    /// `A` down factor, `[rank × in]`. Random, frozen direction. Trains via `a_b`.
    pub a_a: Vec<f32>,
    /// `A` up factor, `[out × rank]`. Zero-initialised (so `A = 0` at start).
    pub a_b: Vec<f32>,
    /// `B` down factor, `[rank × in]`. Random.
    pub b_a: Vec<f32>,
    /// `B` up factor, `[out × rank]`. Zero-initialised (so `B = 0` at start).
    pub b_b: Vec<f32>,
    /// `C` weight-derived bias scale, `[out]`. Zero-initialised.
    pub c: Vec<f32>,
    /// `D` frozen-bias modulation, `[out]`. One-initialised (identity).
    pub d: Vec<f32>,
    /// `E` free additive bias, `[out]`. Zero-initialised.
    pub e: Vec<f32>,
}

impl GloraLinear {
    /// Construct a new GLoRA layer over the supplied frozen `w0` / `b0`.
    ///
    /// All support tensors are initialised to the identity transform.
    ///
    /// # Errors
    /// - [`PeftError::ZeroBlockSize`] if `rank == 0`.
    /// - [`PeftError::RankTooLarge`] if `rank > min(in, out)`.
    /// - [`PeftError::DimensionMismatch`] if `w0.len() != out·in` or `b0.len() != out`.
    pub fn new(
        cfg: &GloraConfig,
        w0: Vec<f32>,
        b0: Vec<f32>,
        rng: &mut LcgRng,
    ) -> PeftResult<Self> {
        if cfg.rank == 0 {
            return Err(PeftError::ZeroBlockSize);
        }
        let upper = cfg.in_features.min(cfg.out_features);
        if cfg.rank > upper {
            return Err(PeftError::RankTooLarge {
                rank: cfg.rank,
                dim: upper,
            });
        }
        if w0.len() != cfg.out_features * cfg.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: cfg.out_features * cfg.in_features,
                got: w0.len(),
            });
        }
        if b0.len() != cfg.out_features {
            return Err(PeftError::DimensionMismatch {
                expected: cfg.out_features,
                got: b0.len(),
            });
        }
        let mut a_a = vec![0.0_f32; cfg.rank * cfg.in_features];
        rng.fill_normal(&mut a_a);
        for v in a_a.iter_mut() {
            *v *= cfg.init_scale;
        }
        let mut b_a = vec![0.0_f32; cfg.rank * cfg.in_features];
        rng.fill_normal(&mut b_a);
        for v in b_a.iter_mut() {
            *v *= cfg.init_scale;
        }
        Ok(Self {
            in_features: cfg.in_features,
            out_features: cfg.out_features,
            rank: cfg.rank,
            w0,
            b0,
            a_a,
            a_b: vec![0.0_f32; cfg.out_features * cfg.rank],
            b_a,
            b_b: vec![0.0_f32; cfg.out_features * cfg.rank],
            c: vec![0.0_f32; cfg.out_features],
            d: vec![1.0_f32; cfg.out_features],
            e: vec![0.0_f32; cfg.out_features],
        })
    }

    /// Materialise the low-rank support tensor `factor_b · factor_a` as a flat `[out × in]`
    /// matrix.
    fn lowrank(&self, factor_b: &[f32], factor_a: &[f32]) -> Vec<f32> {
        let mut m = vec![0.0_f32; self.out_features * self.in_features];
        for i in 0..self.out_features {
            for k in 0..self.rank {
                let b_ik = factor_b[i * self.rank + k];
                for j in 0..self.in_features {
                    m[i * self.in_features + j] += b_ik * factor_a[k * self.in_features + j];
                }
            }
        }
        m
    }

    /// Effective weight `W_eff = W₀ + W₀ ⊙ A + B`.
    #[must_use]
    pub fn effective_weight(&self) -> Vec<f32> {
        let a_mat = self.lowrank(&self.a_b, &self.a_a);
        let b_mat = self.lowrank(&self.b_b, &self.b_a);
        self.w0
            .iter()
            .zip(a_mat.iter())
            .zip(b_mat.iter())
            .map(|((&w0, &a), &b)| w0 + w0 * a + b)
            .collect()
    }

    /// Effective bias `b_eff = C ⊙ rowsum(W₀) + D ⊙ b₀ + E`.
    #[must_use]
    pub fn effective_bias(&self) -> Vec<f32> {
        self.w0
            .chunks(self.in_features.max(1))
            .zip(self.c.iter())
            .zip(self.d.iter())
            .zip(self.b0.iter())
            .zip(self.e.iter())
            .map(|((((row, &c), &d), &b0), &e)| {
                let rowsum: f32 = row.iter().sum();
                c * rowsum + d * b0 + e
            })
            .collect()
    }

    /// Forward pass `y = W_eff·x + b_eff`.
    ///
    /// # Errors
    /// [`PeftError::DimensionMismatch`] if `x.len() != in_features`.
    pub fn forward(&self, x: &[f32]) -> PeftResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let w_eff = self.effective_weight();
        let b_eff = self.effective_bias();
        let mut out = mat_vec(&w_eff, x, self.out_features, self.in_features);
        for (o, &bo) in out.iter_mut().zip(b_eff.iter()) {
            *o += bo;
        }
        Ok(out)
    }

    /// Collapse the GLoRA adapter into a frozen `(W_eff, b_eff)` pair, discarding the support
    /// tensors (reset to identity). Subsequent forwards incur no adapter overhead.
    pub fn merge(&mut self) {
        let w_eff = self.effective_weight();
        let b_eff = self.effective_bias();
        self.w0 = w_eff;
        self.b0 = b_eff;
        for v in self.a_b.iter_mut() {
            *v = 0.0;
        }
        for v in self.b_b.iter_mut() {
            *v = 0.0;
        }
        for v in self.c.iter_mut() {
            *v = 0.0;
        }
        for v in self.d.iter_mut() {
            *v = 1.0;
        }
        for v in self.e.iter_mut() {
            *v = 0.0;
        }
    }

    /// Number of trainable parameters: the `A`/`B` up factors plus the `C`, `D`, `E` vectors.
    ///
    /// (The random down factors `a_a`, `b_a` are treated as frozen, matching the GLoRA search.)
    #[must_use]
    pub fn num_trainable(&self) -> usize {
        let lowrank = 2 * self.out_features * self.rank;
        let vectors = 3 * self.out_features;
        lowrank + vectors
    }
}

/// Multiply matrix `m` (`[rows × cols]`, row-major) by vector `v` (length `cols`).
fn mat_vec(m: &[f32], v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|i| {
            let start = i * cols;
            m[start..start + cols]
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| a * b)
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(in_f: usize, out_f: usize, r: usize, seed: u64) -> (GloraLinear, Vec<f32>, Vec<f32>) {
        let mut rng = LcgRng::new(seed);
        // Non-trivial frozen weight/bias.
        let w0: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let b0: Vec<f32> = (0..out_f).map(|i| (i as f32) * 0.05).collect();
        let cfg = GloraConfig {
            in_features: in_f,
            out_features: out_f,
            rank: r,
            init_scale: 0.05,
        };
        let layer = GloraLinear::new(&cfg, w0.clone(), b0.clone(), &mut rng)
            .expect("GloraLinear::new should succeed with valid config and dimensions");
        (layer, w0, b0)
    }

    #[test]
    fn identity_init_matches_frozen_layer() {
        let (layer, w0, b0) = make(6, 4, 2, 1);
        let x: Vec<f32> = (0..6).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let out = layer
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        // Reference: W₀·x + b₀.
        let mut reference = mat_vec(&w0, &x, 4, 6);
        for (r, &b) in reference.iter_mut().zip(b0.iter()) {
            *r += b;
        }
        for (o, r) in out.iter().zip(reference.iter()) {
            assert!(
                (o - r).abs() < 1e-5,
                "identity init must match frozen: {o} vs {r}"
            );
        }
    }

    #[test]
    fn new_zero_rank_errors() {
        let mut rng = LcgRng::new(2);
        let cfg = GloraConfig {
            in_features: 8,
            out_features: 8,
            rank: 0,
            init_scale: 0.05,
        };
        assert!(matches!(
            GloraLinear::new(&cfg, vec![0.0; 64], vec![0.0; 8], &mut rng),
            Err(PeftError::ZeroBlockSize)
        ));
    }

    #[test]
    fn new_rank_too_large_errors() {
        let mut rng = LcgRng::new(3);
        let cfg = GloraConfig {
            in_features: 4,
            out_features: 8,
            rank: 6,
            init_scale: 0.05,
        };
        assert!(matches!(
            GloraLinear::new(&cfg, vec![0.0; 32], vec![0.0; 8], &mut rng),
            Err(PeftError::RankTooLarge { .. })
        ));
    }

    #[test]
    fn new_w0_dim_mismatch_errors() {
        let mut rng = LcgRng::new(4);
        let cfg = GloraConfig {
            in_features: 8,
            out_features: 8,
            rank: 2,
            init_scale: 0.05,
        };
        assert!(matches!(
            GloraLinear::new(&cfg, vec![0.0; 10], vec![0.0; 8], &mut rng),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn new_b0_dim_mismatch_errors() {
        let mut rng = LcgRng::new(5);
        let cfg = GloraConfig {
            in_features: 8,
            out_features: 8,
            rank: 2,
            init_scale: 0.05,
        };
        assert!(matches!(
            GloraLinear::new(&cfg, vec![0.0; 64], vec![0.0; 3], &mut rng),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_dimension_mismatch_errors() {
        let (layer, _, _) = make(6, 4, 2, 6);
        assert!(matches!(
            layer.forward(&[1.0, 2.0]),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn e_bias_shifts_output() {
        let (mut layer, w0, b0) = make(5, 3, 2, 7);
        layer.e = vec![1.0_f32, 2.0, 3.0];
        let x: Vec<f32> = (0..5).map(|i| (i as f32) * 0.1).collect();
        let out = layer
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        let mut reference = mat_vec(&w0, &x, 3, 5);
        for (i, r) in reference.iter_mut().enumerate() {
            *r += b0[i] + layer.e[i];
        }
        for (o, r) in out.iter().zip(reference.iter()) {
            assert!((o - r).abs() < 1e-5, "E bias must shift output: {o} vs {r}");
        }
    }

    #[test]
    fn d_modulates_frozen_bias() {
        let (mut layer, _, b0) = make(4, 3, 2, 8);
        // Zero the weight so only the bias path matters.
        for v in layer.w0.iter_mut() {
            *v = 0.0;
        }
        layer.d = vec![2.0_f32, 0.5, 0.0];
        let x = vec![0.0_f32; 4]; // zero input → output is pure bias
        let out = layer
            .forward(&x)
            .expect("forward pass should succeed with zero input");
        for o in 0..3 {
            let want = layer.d[o] * b0[o];
            assert!(
                (out[o] - want).abs() < 1e-5,
                "D·b₀ mismatch at {o}: {} vs {want}",
                out[o]
            );
        }
    }

    #[test]
    fn b_lowrank_acts_as_lora_delta() {
        let (mut layer, w0, b0) = make(6, 4, 2, 9);
        // Set B up factor to non-zero → additive weight delta B = b_b·b_a.
        for (i, v) in layer.b_b.iter_mut().enumerate() {
            *v = (i as f32) * 0.1;
        }
        let w_eff = layer.effective_weight();
        let b_mat = layer.lowrank(&layer.b_b, &layer.b_a);
        for idx in 0..w_eff.len() {
            // a_b is zero so the A·W₀ term vanishes; W_eff = W₀ + B.
            let want = w0[idx] + b_mat[idx];
            assert!(
                (w_eff[idx] - want).abs() < 1e-5,
                "B delta mismatch at {idx}"
            );
        }
        let _ = b0;
    }

    #[test]
    fn a_scales_pretrained_weight() {
        let (mut layer, w0, _) = make(6, 4, 2, 10);
        // a_b non-zero → multiplicative modulation W₀ ⊙ A.
        for (i, v) in layer.a_b.iter_mut().enumerate() {
            *v = (i as f32) * 0.05;
        }
        let w_eff = layer.effective_weight();
        let a_mat = layer.lowrank(&layer.a_b, &layer.a_a);
        for idx in 0..w_eff.len() {
            let want = w0[idx] + w0[idx] * a_mat[idx];
            assert!(
                (w_eff[idx] - want).abs() < 1e-5,
                "A scaling mismatch at {idx}"
            );
        }
    }

    #[test]
    fn merge_preserves_function() {
        let (mut layer, _, _) = make(6, 4, 2, 11);
        // Populate every support tensor non-trivially.
        for (i, v) in layer.a_b.iter_mut().enumerate() {
            *v = (i as f32) * 0.02;
        }
        for (i, v) in layer.b_b.iter_mut().enumerate() {
            *v = (i as f32) * 0.03;
        }
        layer.c = vec![0.1, -0.1, 0.2, 0.0];
        layer.d = vec![1.5, 0.5, 1.0, 2.0];
        layer.e = vec![0.3, -0.2, 0.1, 0.0];
        let x: Vec<f32> = (0..6).map(|i| (i as f32) * 0.1 - 0.2).collect();
        let before = layer
            .forward(&x)
            .expect("forward pass should succeed before merge");
        layer.merge();
        let after = layer
            .forward(&x)
            .expect("forward pass should succeed after merge");
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(
                (b - a).abs() < 1e-4,
                "merge must preserve function: {b} vs {a}"
            );
        }
    }

    #[test]
    fn merge_resets_support_tensors() {
        let (mut layer, _, _) = make(6, 4, 2, 12);
        for (i, v) in layer.b_b.iter_mut().enumerate() {
            *v = (i as f32) * 0.1;
        }
        layer.merge();
        assert!(layer.a_b.iter().all(|&v| v == 0.0), "a_b must reset to 0");
        assert!(layer.b_b.iter().all(|&v| v == 0.0), "b_b must reset to 0");
        assert!(layer.c.iter().all(|&v| v == 0.0), "c must reset to 0");
        assert!(layer.d.iter().all(|&v| v == 1.0), "d must reset to 1");
        assert!(layer.e.iter().all(|&v| v == 0.0), "e must reset to 0");
    }

    #[test]
    fn effective_bias_uses_rowsum_for_c() {
        let (mut layer, w0, b0) = make(5, 3, 2, 13);
        layer.c = vec![1.0_f32, 0.0, -1.0];
        layer.d = vec![1.0_f32; 3];
        layer.e = vec![0.0_f32; 3];
        let b_eff = layer.effective_bias();
        for o in 0..3 {
            let rowsum: f32 = w0[o * 5..(o + 1) * 5].iter().sum();
            let want = layer.c[o] * rowsum + b0[o];
            assert!(
                (b_eff[o] - want).abs() < 1e-5,
                "C rowsum bias mismatch at {o}"
            );
        }
    }

    #[test]
    fn num_trainable_counts_factors_and_vectors() {
        let (layer, _, _) = make(8, 6, 3, 14);
        // 2·out·rank + 3·out = 2·6·3 + 3·6 = 36 + 18 = 54
        assert_eq!(layer.num_trainable(), 54);
    }

    #[test]
    fn effective_weight_identity_equals_w0() {
        let (layer, w0, _) = make(6, 4, 2, 15);
        let w_eff = layer.effective_weight();
        for (we, w) in w_eff.iter().zip(w0.iter()) {
            assert!((we - w).abs() < 1e-6, "identity W_eff must equal W₀");
        }
    }
}
