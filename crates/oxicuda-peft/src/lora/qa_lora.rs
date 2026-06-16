use crate::error::{PeftError, PeftResult};
use crate::handle::PeftHandle;
use crate::lora::lora::mat_vec_mul;
use crate::lora::qlora::{nf4_dequantize, quantize_block};

/// Configuration for a QA-LoRA quantization-aware low-rank adapter (Xu et al. 2023 ICLR).
///
/// Key idea: group-wise NF4 quantization with group-specific rank-(rank/n_groups) adapters,
/// allowing each quantization group its own compensation for quantization error.
#[derive(Debug, Clone)]
pub struct QaLoraConfig {
    /// Input dimension; must be divisible by `n_groups`.
    pub in_dim: usize,
    /// Output dimension; must be ≥ 1.
    pub out_dim: usize,
    /// Total LoRA rank; must be divisible by `n_groups` and ≥ `n_groups`.
    pub rank: usize,
    /// Number of quantization groups; must divide `in_dim` and `rank`, ≥ 1.
    pub n_groups: usize,
    /// LoRA scaling factor α.
    pub lora_alpha: f32,
}

/// Quantization-Aware Low-Rank Adaptation layer.
///
/// Stores the dequantized base weight (NF4 round-trip applied) alongside per-group LoRA
/// adapters, where each group has rank `rank / n_groups` and covers `in_dim / n_groups`
/// input columns, allowing fine-grained compensation for group-wise quantization error.
#[derive(Debug, Clone)]
pub struct QaLoraLayer {
    /// Dequantized base weight: `out_dim × in_dim`, row-major (NF4 round-trip applied).
    pub base_weight: Vec<f32>,
    /// Per-group absolute-maximum quantization scales; length `n_groups`.
    pub group_scales: Vec<f32>,
    /// Group-wise LoRA A matrices: `lora_a[g]` is `rank_g × group_in`, row-major.
    pub lora_a: Vec<Vec<f32>>,
    /// Group-wise LoRA B matrices: `lora_b[g]` is `out_dim × rank_g`, row-major.
    pub lora_b: Vec<Vec<f32>>,
    /// Configuration used to construct this layer.
    pub cfg: QaLoraConfig,
}

impl QaLoraLayer {
    /// Construct a `QaLoraLayer` from a pre-existing weight matrix.
    ///
    /// # Errors
    /// Returns `Err` when any dimension constraint is violated or `base_w` has the wrong length.
    pub fn new(cfg: QaLoraConfig, base_w: &[f32], handle: &mut PeftHandle) -> PeftResult<Self> {
        // Validate configuration.
        if cfg.out_dim == 0 {
            return Err(PeftError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if cfg.n_groups == 0 {
            return Err(PeftError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if !cfg.in_dim.is_multiple_of(cfg.n_groups) {
            return Err(PeftError::UnalignedDimension {
                bot: cfg.n_groups,
                in_dim: cfg.in_dim,
            });
        }
        if cfg.rank < cfg.n_groups || !cfg.rank.is_multiple_of(cfg.n_groups) {
            return Err(PeftError::UnalignedDimension {
                bot: cfg.n_groups,
                in_dim: cfg.rank,
            });
        }
        let expected_len = cfg.out_dim * cfg.in_dim;
        if base_w.len() != expected_len {
            return Err(PeftError::DimensionMismatch {
                expected: expected_len,
                got: base_w.len(),
            });
        }

        let rank_g = cfg.rank / cfg.n_groups;
        let group_in = cfg.in_dim / cfg.n_groups;

        // Build base_weight via NF4 round-trip quantization per column group.
        let mut base_weight = base_w.to_vec();
        let mut group_scales = vec![0.0_f32; cfg.n_groups];

        for (g, scale_slot) in group_scales.iter_mut().enumerate() {
            let col_start = g * group_in;
            let col_end = col_start + group_in;

            // Extract the sub-matrix: rows = 0..out_dim, cols = col_start..col_end (row-major).
            let mut block = Vec::with_capacity(cfg.out_dim * group_in);
            for row in 0..cfg.out_dim {
                for col in col_start..col_end {
                    block.push(base_w[row * cfg.in_dim + col]);
                }
            }

            let (codes, absmax) = quantize_block(&block);
            *scale_slot = absmax;

            // Write dequantized values back.
            for (k, &code) in codes.iter().enumerate() {
                let row = k / group_in;
                let col_offset = k % group_in;
                let col = col_start + col_offset;
                base_weight[row * cfg.in_dim + col] = nf4_dequantize(code, absmax);
            }
        }

        // Initialise LoRA weights: A with Kaiming uniform, B with zeros.
        let kaiming_bound = (6.0_f32 / group_in as f32).sqrt();
        let mut lora_a = Vec::with_capacity(cfg.n_groups);
        let mut lora_b = Vec::with_capacity(cfg.n_groups);

        for _ in 0..cfg.n_groups {
            let a_size = rank_g * group_in;
            let a: Vec<f32> = (0..a_size)
                .map(|_| {
                    let u = handle.rng.next_f32();
                    (u * 2.0 - 1.0) * kaiming_bound
                })
                .collect();
            let b = vec![0.0_f32; cfg.out_dim * rank_g];
            lora_a.push(a);
            lora_b.push(b);
        }

        Ok(Self {
            base_weight,
            group_scales,
            lora_a,
            lora_b,
            cfg,
        })
    }

    /// Forward pass: `y = (W_base + scaling · Σ_g B_g A_g) x`, processed per sequence position.
    ///
    /// `x` must have length `seq_len * in_dim`. Returns a vector of length `seq_len * out_dim`.
    ///
    /// # Errors
    /// Returns `Err` when `x.len() != seq_len * in_dim`.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> PeftResult<Vec<f32>> {
        let expected = seq_len * self.cfg.in_dim;
        if x.len() != expected {
            return Err(PeftError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let rank_g = self.cfg.rank / self.cfg.n_groups;
        let group_in = self.cfg.in_dim / self.cfg.n_groups;
        let scale = self.scaling();
        let mut output = Vec::with_capacity(seq_len * self.cfg.out_dim);

        for t in 0..seq_len {
            let x_t = &x[t * self.cfg.in_dim..(t + 1) * self.cfg.in_dim];
            let base_out = mat_vec_mul(&self.base_weight, x_t, self.cfg.out_dim, self.cfg.in_dim);
            let mut lora_delta = vec![0.0_f32; self.cfg.out_dim];

            for g in 0..self.cfg.n_groups {
                let x_g = &x_t[g * group_in..(g + 1) * group_in];
                let h_g = mat_vec_mul(&self.lora_a[g], x_g, rank_g, group_in);
                let contrib = mat_vec_mul(&self.lora_b[g], &h_g, self.cfg.out_dim, rank_g);
                for (d, c) in lora_delta.iter_mut().zip(contrib.iter()) {
                    *d += c;
                }
            }

            for i in 0..self.cfg.out_dim {
                output.push(base_out[i] + scale * lora_delta[i]);
            }
        }

        Ok(output)
    }

    /// Compute the merged weight matrix `W_merged = W_base + scaling · Σ_g B_g A_g`.
    ///
    /// Returns a row-major `out_dim × in_dim` matrix.
    #[must_use]
    pub fn merge(&self) -> Vec<f32> {
        let rank_g = self.cfg.rank / self.cfg.n_groups;
        let group_in = self.cfg.in_dim / self.cfg.n_groups;
        let scale = self.scaling();
        let mut merged = self.base_weight.clone();

        for g in 0..self.cfg.n_groups {
            let col_start = g * group_in;
            // Compute delta_g = B_g @ A_g  →  out_dim × group_in
            for row in 0..self.cfg.out_dim {
                for col in 0..group_in {
                    let mut val = 0.0_f32;
                    for k in 0..rank_g {
                        // B_g[row, k] * A_g[k, col]
                        val +=
                            self.lora_b[g][row * rank_g + k] * self.lora_a[g][k * group_in + col];
                    }
                    merged[row * self.cfg.in_dim + col_start + col] += scale * val;
                }
            }
        }

        merged
    }

    /// Total parameter count: base weight elements + LoRA elements.
    #[must_use]
    pub fn total_params(&self) -> usize {
        self.cfg.out_dim * self.cfg.in_dim + self.lora_params()
    }

    /// LoRA-only parameter count: sum of A and B entries across all groups.
    #[must_use]
    pub fn lora_params(&self) -> usize {
        let rank_g = self.cfg.rank / self.cfg.n_groups;
        let group_in = self.cfg.in_dim / self.cfg.n_groups;
        self.cfg.n_groups * rank_g * (group_in + self.cfg.out_dim)
    }

    /// Effective LoRA scaling: `lora_alpha / rank_g`.
    #[must_use]
    pub fn scaling(&self) -> f32 {
        let rank_g = self.cfg.rank / self.cfg.n_groups;
        self.cfg.lora_alpha / rank_g as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::PeftHandle;

    fn make_handle(seed: u64) -> PeftHandle {
        PeftHandle::new(80, seed)
    }

    fn make_cfg(in_dim: usize, out_dim: usize, rank: usize, n_groups: usize) -> QaLoraConfig {
        QaLoraConfig {
            in_dim,
            out_dim,
            rank,
            n_groups,
            lora_alpha: 8.0,
        }
    }

    fn make_layer(cfg: QaLoraConfig, seed: u64) -> PeftResult<QaLoraLayer> {
        let mut h = make_handle(seed);
        let base_w = vec![0.1_f32; cfg.out_dim * cfg.in_dim];
        QaLoraLayer::new(cfg, &base_w, &mut h)
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 1: B is zero-initialised, so forward == W_base @ x.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn zero_b_forward_is_base() {
        let cfg = make_cfg(4, 3, 4, 2);
        let out_dim = cfg.out_dim;
        let in_dim = cfg.in_dim;
        let mut h = make_handle(1);
        let base_w: Vec<f32> = (0..(out_dim * in_dim)).map(|i| i as f32 * 0.01).collect();
        let layer = QaLoraLayer::new(cfg, &base_w, &mut h)
            .expect("QaLoraLayer creation should succeed with valid config");
        // B is all zeros → LoRA delta is zero; output must equal W_base @ x.
        let x = vec![1.0_f32; in_dim];
        let out = layer
            .forward(&x, 1)
            .expect("forward pass should succeed with valid input");
        let base_out = mat_vec_mul(&layer.base_weight, &x, out_dim, in_dim);
        for (a, b) in out.iter().zip(base_out.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 2: merge() returns a vector of length out_dim * in_dim.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn merge_shape() {
        let cfg = make_cfg(4, 3, 4, 2);
        let out_dim = cfg.out_dim;
        let in_dim = cfg.in_dim;
        let layer = make_layer(cfg, 2).expect("layer creation should succeed for merge_shape test");
        assert_eq!(layer.merge().len(), out_dim * in_dim);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 3: lora_params() matches the analytical formula.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn lora_params_formula() {
        let n_groups = 2usize;
        let rank = 4usize;
        let in_dim = 4usize;
        let out_dim = 3usize;
        let cfg = make_cfg(in_dim, out_dim, rank, n_groups);
        let layer =
            make_layer(cfg, 3).expect("layer creation should succeed for lora_params_formula test");
        let rank_g = rank / n_groups;
        let group_in = in_dim / n_groups;
        assert_eq!(
            layer.lora_params(),
            n_groups * rank_g * (group_in + out_dim)
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 4: total_params() == base + lora.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn total_params_formula() {
        let cfg = make_cfg(4, 3, 4, 2);
        let layer = make_layer(cfg, 4)
            .expect("layer creation should succeed for total_params_formula test");
        assert_eq!(
            layer.total_params(),
            layer.cfg.out_dim * layer.cfg.in_dim + layer.lora_params()
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 5: scaling() == lora_alpha / rank_g.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn scaling_value() {
        let n_groups = 2usize;
        let rank = 4usize;
        let cfg = make_cfg(4, 3, rank, n_groups);
        let alpha = cfg.lora_alpha;
        let layer =
            make_layer(cfg, 5).expect("layer creation should succeed for scaling_value test");
        let rank_g = rank / n_groups;
        assert!((layer.scaling() - alpha / rank_g as f32).abs() < 1e-7);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 6: group_scales has length n_groups.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn group_scales_len() {
        let n_groups = 3usize;
        let cfg = make_cfg(6, 4, 6, n_groups);
        let layer =
            make_layer(cfg, 6).expect("layer creation should succeed for group_scales_len test");
        assert_eq!(layer.group_scales.len(), n_groups);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 7: base_weight has length out_dim * in_dim.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn base_weight_len() {
        let cfg = make_cfg(4, 3, 4, 2);
        let layer =
            make_layer(cfg, 7).expect("layer creation should succeed for base_weight_len test");
        assert_eq!(
            layer.base_weight.len(),
            layer.cfg.out_dim * layer.cfg.in_dim
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 8: forward with seq_len=3 returns vec of length 3*out_dim.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn seq_len_gt_1() {
        let cfg = make_cfg(4, 3, 4, 2);
        let out_dim = cfg.out_dim;
        let in_dim = cfg.in_dim;
        let layer =
            make_layer(cfg, 8).expect("layer creation should succeed for seq_len_gt_1 test");
        let x = vec![0.5_f32; 3 * in_dim];
        let out = layer
            .forward(&x, 3)
            .expect("forward pass should succeed with seq_len=3");
        assert_eq!(out.len(), 3 * out_dim);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 9: Kaiming uniform range check for lora_a[0].
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn kaiming_range() {
        let in_dim = 4usize;
        let n_groups = 2usize;
        let group_in = in_dim / n_groups;
        let cfg = make_cfg(in_dim, 3, 4, n_groups);
        let layer =
            make_layer(cfg, 9).expect("layer creation should succeed for kaiming_range test");
        let bound = (6.0_f32 / group_in as f32).sqrt() + 1e-5;
        for &v in &layer.lora_a[0] {
            assert!(v.abs() <= bound, "value {v} out of Kaiming range ±{bound}");
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 10: base_weight entries are quantized NF4 values (verify round-trip happened).
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn base_quantized() {
        use crate::lora::qlora::NF4_TABLE;
        let cfg = make_cfg(4, 3, 4, 2);
        let mut h = make_handle(10);
        let base_w: Vec<f32> = (0..12).map(|i| i as f32 * 0.15 - 0.5).collect();
        let layer = QaLoraLayer::new(cfg, &base_w, &mut h)
            .expect("QaLoraLayer creation should succeed with valid base weights");
        // Every element in base_weight must be absmax * NF4_TABLE[some_idx] for its group.
        // Check that each value is one of the 16 NF4 table values times some absmax.
        for &bw in &layer.base_weight {
            let matched = layer
                .group_scales
                .iter()
                .any(|&absmax| NF4_TABLE.iter().any(|&t| (bw - t * absmax).abs() < 1e-5));
            assert!(
                matched,
                "base_weight value {bw} is not a valid NF4 dequantized value"
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 11: in_dim not divisible by n_groups → Err.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn err_in_dim_not_divisible() {
        let cfg = QaLoraConfig {
            in_dim: 5,
            out_dim: 3,
            rank: 6,
            n_groups: 3,
            lora_alpha: 1.0,
        };
        let mut h = make_handle(11);
        let base_w = vec![0.0_f32; 15];
        assert!(QaLoraLayer::new(cfg, &base_w, &mut h).is_err());
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 12: rank not divisible by n_groups → Err.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn err_rank_not_divisible() {
        let cfg = QaLoraConfig {
            in_dim: 6,
            out_dim: 3,
            rank: 5,
            n_groups: 3,
            lora_alpha: 1.0,
        };
        let mut h = make_handle(12);
        let base_w = vec![0.0_f32; 18];
        assert!(QaLoraLayer::new(cfg, &base_w, &mut h).is_err());
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 13: base_w wrong length → Err.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn err_wrong_base_len() {
        let cfg = make_cfg(4, 3, 4, 2);
        let mut h = make_handle(13);
        let base_w = vec![0.0_f32; 5]; // wrong
        assert!(QaLoraLayer::new(cfg, &base_w, &mut h).is_err());
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 14: same seed → same forward output (determinism).
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn deterministic() {
        let cfg_a = make_cfg(4, 3, 4, 2);
        let cfg_b = make_cfg(4, 3, 4, 2);
        let in_dim = cfg_a.in_dim;
        let base_w: Vec<f32> = (0..12).map(|i| i as f32 * 0.05).collect();
        let mut h_a = make_handle(42);
        let mut h_b = make_handle(42);
        let layer_a = QaLoraLayer::new(cfg_a, &base_w, &mut h_a)
            .expect("layer_a creation should succeed with valid config");
        let layer_b = QaLoraLayer::new(cfg_b, &base_w, &mut h_b)
            .expect("layer_b creation should succeed with valid config");
        let x = vec![0.3_f32; in_dim];
        let out_a = layer_a
            .forward(&x, 1)
            .expect("layer_a forward should succeed");
        let out_b = layer_b
            .forward(&x, 1)
            .expect("layer_b forward should succeed");
        assert_eq!(out_a, out_b);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 15: manually set lora_b[0][0]=1, lora_a[0][0]=1 → LoRA contributes.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn lora_contribution() {
        let cfg = make_cfg(4, 3, 4, 2);
        let in_dim = cfg.in_dim;
        let mut h = make_handle(15);
        let base_w = vec![0.0_f32; cfg.out_dim * cfg.in_dim];
        let mut layer = QaLoraLayer::new(cfg, &base_w, &mut h)
            .expect("layer creation should succeed for lora_contribution test");
        // Set A[0][0,0]=1, B[0][0,0]=1 → group-0 contributes scaling() to output[0].
        layer.lora_a[0][0] = 1.0;
        layer.lora_b[0][0] = 1.0;
        let mut x = vec![0.0_f32; in_dim];
        x[0] = 1.0; // group-0 input col 0
        let out = layer
            .forward(&x, 1)
            .expect("forward pass should succeed after setting lora weights");
        let expected = layer.scaling();
        assert!(
            (out[0] - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            out[0]
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 16: n_groups=1 degenerates to standard whole-matrix adapter.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn n_groups_1() {
        let cfg = make_cfg(4, 3, 4, 1);
        let in_dim = cfg.in_dim;
        let layer = make_layer(cfg, 16).expect("layer creation should succeed for n_groups_1 test");
        assert_eq!(layer.group_scales.len(), 1);
        let x = vec![1.0_f32; in_dim];
        let out = layer
            .forward(&x, 1)
            .expect("forward pass should succeed with single group");
        assert_eq!(out.len(), 3);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 17: x wrong length → Err.
    // ───────────────────────────────────────────────────────────────────────
    #[test]
    fn forward_dim_err() {
        let cfg = make_cfg(4, 3, 4, 2);
        let layer =
            make_layer(cfg, 17).expect("layer creation should succeed for forward_dim_err test");
        let x = vec![0.0_f32; 5]; // not seq_len * in_dim
        assert!(layer.forward(&x, 1).is_err());
    }
}
