//! LoRA (Low-Rank Adaptation) adapter for linear layers.
//!
//! Implements LoRA as described in Hu et al. 2021:
//! `W' = W_0 + (α/r) * B * A`
//! where `A` is Gaussian-initialised and `B` is zero-initialised.

use std::collections::HashMap;

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;

// ─── LoraConfig ───────────────────────────────────────────────────────────────

/// Configuration for LoRA adaptation.
///
/// # Reference
/// Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Intrinsic rank `r` (must be >= 1).
    pub rank: usize,
    /// Scaling factor `α` (must be > 0).
    pub alpha: f32,
    /// Dropout probability (applied to input before LoRA path).
    pub dropout: f32,
    /// Names of target modules to adapt.
    pub target_modules: Vec<String>,
}

impl LoraConfig {
    /// Create a new LoRA config with the given rank and alpha.
    ///
    /// # Errors
    /// - `InvalidLoraRank` if `rank == 0`
    /// - `InvalidLoraAlpha` if `alpha <= 0`
    pub fn new(rank: usize, alpha: f32) -> GenResult<Self> {
        if rank == 0 {
            return Err(GenError::InvalidLoraRank(rank));
        }
        if alpha <= 0.0 {
            return Err(GenError::InvalidLoraAlpha(alpha));
        }
        Ok(Self {
            rank,
            alpha,
            dropout: 0.0,
            target_modules: Vec::new(),
        })
    }

    /// Create a config with dropout and target modules.
    pub fn with_options(
        rank: usize,
        alpha: f32,
        dropout: f32,
        target_modules: Vec<String>,
    ) -> GenResult<Self> {
        if rank == 0 {
            return Err(GenError::InvalidLoraRank(rank));
        }
        if alpha <= 0.0 {
            return Err(GenError::InvalidLoraAlpha(alpha));
        }
        Ok(Self {
            rank,
            alpha,
            dropout: dropout.clamp(0.0, 1.0),
            target_modules,
        })
    }

    /// Compute the LoRA scaling factor: `α / r`.
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

// ─── LoraLinear ───────────────────────────────────────────────────────────────

/// LoRA adapter for a single linear layer.
///
/// Implements `y = x @ W_0^T + (α/r) * (x @ A^T) @ B^T`
/// where `A: [r × in]` (Gaussian init) and `B: [out × r]` (zero init).
#[derive(Debug, Clone)]
pub struct LoraLinear {
    in_features: usize,
    out_features: usize,
    rank: usize,
    scaling: f32,
    /// A matrix: `[rank × in_features]`, Gaussian-initialised.
    matrix_a: Vec<f32>,
    /// B matrix: `[out_features × rank]`, zero-initialised.
    matrix_b: Vec<f32>,
}

impl LoraLinear {
    /// Create a new LoRA linear adapter.
    ///
    /// # Arguments
    /// - `in_features`: Input dimensionality.
    /// - `out_features`: Output dimensionality.
    /// - `config`: LoRA configuration.
    /// - `rng`: Seeded RNG for Gaussian initialisation of A.
    ///
    /// # Errors
    /// - `EmptyInput` if dimensions are 0
    /// - Inherits errors from `config`
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: &LoraConfig,
        rng: &mut LcgRng,
    ) -> GenResult<Self> {
        if in_features == 0 {
            return Err(GenError::EmptyInput("in_features must be > 0"));
        }
        if out_features == 0 {
            return Err(GenError::EmptyInput("out_features must be > 0"));
        }
        // Initialise A ~ N(0, 1/r) (Gaussian, stddev = 1/sqrt(r))
        let std = 1.0 / (config.rank as f32).sqrt();
        let a_size = config.rank * in_features;
        let mut matrix_a = vec![0.0_f32; a_size];
        // Use Box-Muller transform via LcgRng::fill_normal then scale
        rng.fill_normal(&mut matrix_a);
        for v in &mut matrix_a {
            *v *= std;
        }
        // Initialise B = 0
        let matrix_b = vec![0.0_f32; out_features * config.rank];
        Ok(Self {
            in_features,
            out_features,
            rank: config.rank,
            scaling: config.scaling(),
            matrix_a,
            matrix_b,
        })
    }

    /// Create from pre-computed matrices (for testing / merging).
    pub fn from_matrices(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        matrix_a: Vec<f32>,
        matrix_b: Vec<f32>,
    ) -> GenResult<Self> {
        if matrix_a.len() != rank * in_features {
            return Err(GenError::DimensionMismatch {
                expected: rank * in_features,
                got: matrix_a.len(),
            });
        }
        if matrix_b.len() != out_features * rank {
            return Err(GenError::DimensionMismatch {
                expected: out_features * rank,
                got: matrix_b.len(),
            });
        }
        let scaling = alpha / rank as f32;
        Ok(Self {
            in_features,
            out_features,
            rank,
            scaling,
            matrix_a,
            matrix_b,
        })
    }

    /// Apply LoRA to a batch of inputs.
    ///
    /// `y = base_output + scaling * (x @ A^T) @ B^T`
    ///
    /// # Arguments
    /// - `x`: Input of shape `[batch × in_features]`.
    /// - `base_output`: Pre-computed `x @ W_0^T` of shape `[batch × out_features]`.
    /// - `batch`: Number of input samples.
    ///
    /// # Errors
    /// - `DimensionMismatch` if shapes don't match
    /// - `EmptyInput` if inputs are empty
    pub fn forward(&self, x: &[f32], base_output: &[f32], batch: usize) -> GenResult<Vec<f32>> {
        if x.is_empty() {
            return Err(GenError::EmptyInput("x is empty"));
        }
        let expected_x = batch * self.in_features;
        if x.len() != expected_x {
            return Err(GenError::DimensionMismatch {
                expected: expected_x,
                got: x.len(),
            });
        }
        let expected_base = batch * self.out_features;
        if base_output.len() != expected_base {
            return Err(GenError::DimensionMismatch {
                expected: expected_base,
                got: base_output.len(),
            });
        }
        // Compute x @ A^T: [batch × rank]
        // A: [rank × in_features], so A^T: [in_features × rank]
        // Result[b, r] = sum_i x[b, i] * A[r, i]
        let xa = Self::matmul(x, &self.matrix_a, batch, self.in_features, self.rank);
        // Compute (x @ A^T) @ B^T: [batch × out_features]
        // B: [out_features × rank], so B^T: [rank × out_features]
        // Result[b, o] = sum_r xa[b, r] * B[o, r]
        let xab = Self::matmul(&xa, &self.matrix_b, batch, self.rank, self.out_features);
        // Add to base: y = base + scaling * xab
        let result = base_output
            .iter()
            .zip(&xab)
            .map(|(&b, &d)| b + self.scaling * d)
            .collect();
        Ok(result)
    }

    /// Matrix multiplication: `C = A @ B^T`.
    ///
    /// - `A: [m × k]`, `B: [n × k]`, result `C: [m × n]`.
    /// - `B` is accessed in row-major as `[n × k]`, so `B^T` is `[k × n]`.
    pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for l in 0..k {
                    acc += a[i * k + l] * b[j * k + l];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    /// Compute `B @ A`: the LoRA delta matrix `[out_features × in_features]`.
    ///
    /// This is used for weight merging: `delta = B @ A`.
    pub fn delta_weight(&self) -> Vec<f32> {
        // B: [out × rank], A: [rank × in]
        // B @ A: [out × in]
        // C[o, i] = sum_r B[o, r] * A[r, i]
        let mut delta = vec![0.0_f32; self.out_features * self.in_features];
        for o in 0..self.out_features {
            for i in 0..self.in_features {
                let mut acc = 0.0_f32;
                for r in 0..self.rank {
                    acc +=
                        self.matrix_b[o * self.rank + r] * self.matrix_a[r * self.in_features + i];
                }
                delta[o * self.in_features + i] = acc;
            }
        }
        delta
    }

    // ─── Accessors ──────────────────────────────────────────────────────────

    /// Return the intrinsic rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Return the scaling factor `α/r`.
    pub fn scaling(&self) -> f32 {
        self.scaling
    }

    /// Return the A matrix (read-only).
    pub fn matrix_a(&self) -> &[f32] {
        &self.matrix_a
    }

    /// Return the B matrix (read-only).
    pub fn matrix_b(&self) -> &[f32] {
        &self.matrix_b
    }

    /// Return the B matrix (mutable).
    pub fn matrix_b_mut(&mut self) -> &mut Vec<f32> {
        &mut self.matrix_b
    }

    /// Return the input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Return the output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

// ─── LoraModel ────────────────────────────────────────────────────────────────

/// A model consisting of multiple named LoRA adapters.
///
/// Associates a string name with each `LoraLinear` adapter, allowing
/// per-layer adaptation.
#[derive(Debug, Clone)]
pub struct LoraModel {
    adapters: HashMap<String, LoraLinear>,
    config: LoraConfig,
}

impl LoraModel {
    /// Create an empty LoRA model with the given config.
    pub fn new(config: LoraConfig) -> Self {
        Self {
            adapters: HashMap::new(),
            config,
        }
    }

    /// Add a named LoRA adapter to the model.
    pub fn add_adapter(&mut self, name: impl Into<String>, adapter: LoraLinear) {
        self.adapters.insert(name.into(), adapter);
    }

    /// Get a reference to a named adapter, if it exists.
    pub fn get_adapter(&self, name: &str) -> Option<&LoraLinear> {
        self.adapters.get(name)
    }

    /// Return the total number of adapters.
    pub fn adapter_count(&self) -> usize {
        self.adapters.len()
    }

    /// Return the LoRA config.
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// Iterate over all adapters.
    pub fn adapters(&self) -> &HashMap<String, LoraLinear> {
        &self.adapters
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn lora_config_valid() {
        let config = LoraConfig::new(4, 8.0).unwrap();
        assert_eq!(config.rank, 4);
        assert!((config.alpha - 8.0).abs() < 1e-6);
        assert!((config.scaling() - 2.0).abs() < 1e-6); // 8/4 = 2
    }

    #[test]
    fn lora_config_invalid_rank() {
        assert!(matches!(
            LoraConfig::new(0, 8.0),
            Err(GenError::InvalidLoraRank(0))
        ));
    }

    #[test]
    fn lora_config_invalid_alpha() {
        assert!(matches!(
            LoraConfig::new(4, 0.0),
            Err(GenError::InvalidLoraAlpha(_))
        ));
        assert!(matches!(
            LoraConfig::new(4, -1.0),
            Err(GenError::InvalidLoraAlpha(_))
        ));
    }

    #[test]
    fn lora_linear_new_valid() {
        let config = LoraConfig::new(4, 4.0).unwrap();
        let mut rng = make_rng();
        let lora = LoraLinear::new(16, 32, &config, &mut rng).unwrap();
        assert_eq!(lora.in_features(), 16);
        assert_eq!(lora.out_features(), 32);
        assert_eq!(lora.rank(), 4);
        assert_eq!(lora.matrix_a().len(), 4 * 16);
        assert_eq!(lora.matrix_b().len(), 32 * 4);
    }

    #[test]
    fn lora_b_init_is_zero() {
        let config = LoraConfig::new(4, 4.0).unwrap();
        let mut rng = make_rng();
        let lora = LoraLinear::new(16, 32, &config, &mut rng).unwrap();
        // B should be zero-initialised
        for &v in lora.matrix_b() {
            assert_eq!(v, 0.0, "B matrix should be zero-initialised");
        }
    }

    #[test]
    fn lora_zero_b_gives_base_output() {
        let config = LoraConfig::new(4, 4.0).unwrap();
        let mut rng = make_rng();
        let lora = LoraLinear::new(8, 16, &config, &mut rng).unwrap();
        // With B=0, forward should return base_output unchanged
        let x = vec![1.0_f32; 8]; // batch=1
        let base_output = vec![0.5_f32; 16];
        let out = lora.forward(&x, &base_output, 1).unwrap();
        for (&o, &b) in out.iter().zip(&base_output) {
            assert!(
                (o - b).abs() < 1e-5,
                "B=0 should not change base output: {o} vs {b}"
            );
        }
    }

    #[test]
    fn lora_forward_output_shape() {
        let config = LoraConfig::new(2, 2.0).unwrap();
        let mut rng = make_rng();
        let mut lora = LoraLinear::new(8, 16, &config, &mut rng).unwrap();
        // Set B to something nonzero
        for v in lora.matrix_b_mut() {
            *v = 0.1;
        }
        let x = vec![1.0_f32; 3 * 8]; // batch=3
        let base = vec![0.0_f32; 3 * 16];
        let out = lora.forward(&x, &base, 3).unwrap();
        assert_eq!(out.len(), 3 * 16);
    }

    #[test]
    fn matmul_correctness() {
        // 2×2 @ 2×2^T = 2×2
        // A = [[1,0],[0,1]], B = [[1,0],[0,1]] → C = A @ B^T = I @ I = I
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let b = vec![1.0_f32, 0.0, 0.0, 1.0];
        let c = LoraLinear::matmul(&a, &b, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-5, "c[0]={}", c[0]);
        assert!((c[1] - 0.0).abs() < 1e-5, "c[1]={}", c[1]);
        assert!((c[2] - 0.0).abs() < 1e-5, "c[2]={}", c[2]);
        assert!((c[3] - 1.0).abs() < 1e-5, "c[3]={}", c[3]);
    }

    #[test]
    fn lora_scaling_correctness() {
        // If A = e_1^T (picks first column), B = e_1 (writes to first output),
        // and input x = [1, 0, ...], then delta = B@A picks x[0] and writes to out[0]
        let rank = 1;
        let in_f = 4;
        let out_f = 4;
        let alpha = 2.0;
        let mut a = vec![0.0_f32; rank * in_f];
        a[0] = 1.0; // A = [[1, 0, 0, 0]]
        let mut b = vec![0.0_f32; out_f * rank];
        b[0] = 1.0; // B = [[1], [0], [0], [0]]
        let lora = LoraLinear::from_matrices(in_f, out_f, rank, alpha, a, b).unwrap();
        let x = vec![1.0_f32, 0.0, 0.0, 0.0];
        let base = vec![0.0_f32; out_f];
        let out = lora.forward(&x, &base, 1).unwrap();
        // Expected: scaling * x[0] = (alpha/rank) * 1 = 2 * 1 = 2
        assert!((out[0] - 2.0).abs() < 1e-5, "expected 2.0, got {}", out[0]);
        // All other outputs should be 0
        for &v in &out[1..] {
            assert!(v.abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn lora_model_add_and_get() {
        let config = LoraConfig::new(4, 4.0).unwrap();
        let mut model = LoraModel::new(config.clone());
        let mut rng = make_rng();
        let adapter = LoraLinear::new(16, 32, &config, &mut rng).unwrap();
        model.add_adapter("attn.q_proj", adapter);
        assert_eq!(model.adapter_count(), 1);
        assert!(model.get_adapter("attn.q_proj").is_some());
        assert!(model.get_adapter("nonexistent").is_none());
    }

    #[test]
    fn delta_weight_shape() {
        let config = LoraConfig::new(4, 4.0).unwrap();
        let mut rng = make_rng();
        let lora = LoraLinear::new(8, 16, &config, &mut rng).unwrap();
        let delta = lora.delta_weight();
        assert_eq!(delta.len(), 16 * 8);
        // B=0 → delta should be all zeros
        for &v in &delta {
            assert_eq!(v, 0.0, "delta should be 0 when B=0");
        }
    }

    #[test]
    fn lora_a_not_zero() {
        // A should NOT be zero after Gaussian init
        let config = LoraConfig::new(4, 4.0).unwrap();
        let mut rng = make_rng();
        let lora = LoraLinear::new(16, 32, &config, &mut rng).unwrap();
        let sum: f32 = lora.matrix_a().iter().map(|v| v.abs()).sum();
        assert!(
            sum > 1e-5,
            "A should be nonzero after Gaussian init: sum={sum}"
        );
    }
}
