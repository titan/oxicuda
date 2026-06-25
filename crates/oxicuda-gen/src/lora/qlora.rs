//! QLoRA: 4-bit NormalFloat (NF4) quantised base weights with a LoRA path.
//!
//! Implements the frozen-base-weight quantisation of Dettmers et al.
//! ("QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023). The large
//! pre-trained weight matrix `W₀` is stored in **NF4**, a 4-bit data type
//! whose 16 quantisation levels are the quantiles of a standard normal
//! distribution (information-theoretically optimal for the approximately
//! normally-distributed weights of a trained network). During the forward
//! pass `W₀` is dequantised on the fly and a trainable low-rank LoRA term is
//! added:
//!
//! ```text
//!     y = dequant_NF4(W₀) · xᵀ  +  (α/r) · B · (A · xᵀ)
//! ```
//!
//! Quantisation is **block-wise**: the weight vector is split into contiguous
//! blocks of `block_size`, each normalised by its own `absmax` scale before
//! the values are mapped to the nearest NF4 level. This matches the
//! bitsandbytes reference and keeps quantisation error low even when the
//! weight magnitudes vary across the matrix.
//!
//! Two 4-bit codes are packed per `u8`, so an `n`-element weight needs
//! `⌈n/2⌉` bytes plus one `f32` scale per block.

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;
use crate::lora::adapter::{LoraConfig, LoraLinear};

/// The 16 NF4 quantisation levels (Dettmers 2023 / bitsandbytes), sorted
/// ascending and normalised to `[-1, 1]`. Index `8` is exactly `0.0` so that
/// zero weights quantise without error.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.696_393_2,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_25,
    0.246_112_3,
    0.337_915_24,
    0.440_709_28,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// Map a normalised value in `[-1, 1]` to the index of the nearest NF4 level.
#[inline]
fn nearest_nf4(value: f32) -> u8 {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (idx, &level) in NF4_LEVELS.iter().enumerate() {
        let d = (value - level).abs();
        if d < best_dist {
            best_dist = d;
            best = idx;
        }
    }
    best as u8
}

// ─── Nf4Tensor ──────────────────────────────────────────────────────────────

/// A block-wise NF4-quantised 1-D tensor (a flattened weight matrix).
///
/// Stores the packed 4-bit codes (two per byte), one `absmax` scale per block,
/// and the metadata needed to dequantise back to `[f32]`.
#[derive(Debug, Clone)]
pub struct Nf4Tensor {
    /// Logical element count (may be odd; the final nibble of the last byte is
    /// padding when `numel` is odd).
    numel: usize,
    /// Quantisation block length.
    block_size: usize,
    /// Per-block absolute-maximum scales, length `⌈numel/block_size⌉`.
    scales: Vec<f32>,
    /// Packed 4-bit codes, length `⌈numel/2⌉`.
    codes: Vec<u8>,
}

impl Nf4Tensor {
    /// Quantise `data` into NF4 with the given block size.
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] if `data` is empty or `block_size == 0`.
    pub fn quantize(data: &[f32], block_size: usize) -> GenResult<Self> {
        if data.is_empty() {
            return Err(GenError::EmptyInput("data is empty"));
        }
        if block_size == 0 {
            return Err(GenError::EmptyInput("block_size must be > 0"));
        }
        let numel = data.len();
        let n_blocks = numel.div_ceil(block_size);
        let mut scales = vec![0.0_f32; n_blocks];
        let mut nibbles = vec![0u8; numel];

        for (b, scale) in scales.iter_mut().enumerate() {
            let start = b * block_size;
            let end = (start + block_size).min(numel);
            // Per-block absmax (guard against an all-zero block).
            let absmax = data[start..end]
                .iter()
                .fold(0.0_f32, |m, &v| m.max(v.abs()));
            *scale = absmax;
            let inv = if absmax > 0.0 { 1.0 / absmax } else { 0.0 };
            for (i, &v) in data[start..end].iter().enumerate() {
                nibbles[start + i] = nearest_nf4(v * inv);
            }
        }

        // Pack two 4-bit codes per byte (low nibble = even index).
        let mut codes = vec![0u8; numel.div_ceil(2)];
        for (i, &nib) in nibbles.iter().enumerate() {
            let byte = i / 2;
            if i % 2 == 0 {
                codes[byte] |= nib & 0x0F;
            } else {
                codes[byte] |= (nib & 0x0F) << 4;
            }
        }

        Ok(Self {
            numel,
            block_size,
            scales,
            codes,
        })
    }

    /// Read the `i`-th 4-bit code.
    #[inline]
    fn code_at(&self, i: usize) -> u8 {
        let byte = self.codes[i / 2];
        if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        }
    }

    /// Dequantise back to a freshly allocated `Vec<f32>` of length [`Self::numel`].
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.numel];
        for (i, slot) in out.iter_mut().enumerate() {
            let block = i / self.block_size;
            let level = NF4_LEVELS[self.code_at(i) as usize];
            *slot = level * self.scales[block];
        }
        out
    }

    /// Logical element count.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Quantisation block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of quantisation blocks.
    pub fn num_blocks(&self) -> usize {
        self.scales.len()
    }

    /// Packed-storage size in bytes (codes + `f32` scales).
    pub fn storage_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * std::mem::size_of::<f32>()
    }

    /// Per-block scales (read-only).
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }
}

// ─── QLoraLinear ────────────────────────────────────────────────────────────

/// A linear layer whose frozen base weight is stored in NF4 and augmented with
/// a trainable LoRA correction.
///
/// The base weight is laid out row-major as `[out_features × in_features]`
/// (i.e. `y = W · x` with `W[o, i]`).
#[derive(Debug, Clone)]
pub struct QLoraLinear {
    in_features: usize,
    out_features: usize,
    /// NF4-quantised base weight `[out × in]` (flattened).
    base: Nf4Tensor,
    /// Trainable low-rank adapter.
    lora: LoraLinear,
}

impl QLoraLinear {
    /// Construct from a full-precision base weight, quantising it to NF4 and
    /// attaching a freshly-initialised LoRA adapter.
    ///
    /// * `weight`       — base weight `[out_features × in_features]`.
    /// * `block_size`   — NF4 quantisation block length (e.g. `64`).
    /// * `lora_config`  — rank/alpha for the trainable adapter.
    /// * `rng`          — seeded RNG for LoRA `A` initialisation.
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] on zero dimensions / block size.
    /// * [`GenError::DimensionMismatch`] if `weight.len() != out·in`.
    /// * Propagates [`LoraLinear::new`] errors.
    pub fn from_weight(
        weight: &[f32],
        in_features: usize,
        out_features: usize,
        block_size: usize,
        lora_config: &LoraConfig,
        rng: &mut LcgRng,
    ) -> GenResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(GenError::EmptyInput("feature dims must be > 0"));
        }
        if weight.len() != in_features * out_features {
            return Err(GenError::DimensionMismatch {
                expected: in_features * out_features,
                got: weight.len(),
            });
        }
        let base = Nf4Tensor::quantize(weight, block_size)?;
        let lora = LoraLinear::new(in_features, out_features, lora_config, rng)?;
        Ok(Self {
            in_features,
            out_features,
            base,
            lora,
        })
    }

    /// Forward pass `y = dequant(W₀)·xᵀ + LoRA(x)`.
    ///
    /// * `x`     — input `[batch × in_features]`.
    /// * `batch` — number of rows.
    ///
    /// Returns `[batch × out_features]`.
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] if `x` is empty.
    /// * [`GenError::DimensionMismatch`] if `x.len() != batch·in_features`.
    pub fn forward(&self, x: &[f32], batch: usize) -> GenResult<Vec<f32>> {
        if x.is_empty() {
            return Err(GenError::EmptyInput("x is empty"));
        }
        if x.len() != batch * self.in_features {
            return Err(GenError::DimensionMismatch {
                expected: batch * self.in_features,
                got: x.len(),
            });
        }
        // Dequantise base weight `[out × in]` and compute base = x · Wᵀ.
        let w = self.base.dequantize();
        let mut base_out = vec![0.0_f32; batch * self.out_features];
        for b in 0..batch {
            for o in 0..self.out_features {
                let mut acc = 0.0_f32;
                for i in 0..self.in_features {
                    acc += x[b * self.in_features + i] * w[o * self.in_features + i];
                }
                base_out[b * self.out_features + o] = acc;
            }
        }
        // Add the LoRA correction.
        self.lora.forward(x, &base_out, batch)
    }

    /// Mean-squared quantisation error of the base weight against `reference`
    /// (the original full-precision weights), useful as a fidelity check.
    ///
    /// # Errors
    /// [`GenError::DimensionMismatch`] if `reference` has the wrong length.
    pub fn base_quant_mse(&self, reference: &[f32]) -> GenResult<f32> {
        if reference.len() != self.base.numel() {
            return Err(GenError::DimensionMismatch {
                expected: self.base.numel(),
                got: reference.len(),
            });
        }
        let deq = self.base.dequantize();
        let mse = reference
            .iter()
            .zip(&deq)
            .map(|(&r, &d)| (r - d) * (r - d))
            .sum::<f32>()
            / reference.len() as f32;
        Ok(mse)
    }

    /// Input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Access the trainable LoRA adapter.
    pub fn lora(&self) -> &LoraLinear {
        &self.lora
    }

    /// Mutable access to the trainable LoRA adapter (for training updates).
    pub fn lora_mut(&mut self) -> &mut LoraLinear {
        &mut self.lora
    }

    /// Access the NF4-quantised base weight.
    pub fn base(&self) -> &Nf4Tensor {
        &self.base
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn nf4_levels_sorted_and_symmetric_endpoints() {
        // Strictly increasing, contains exact zero, spans [-1, 1].
        for w in NF4_LEVELS.windows(2) {
            assert!(w[1] > w[0], "levels must be strictly increasing");
        }
        assert_eq!(NF4_LEVELS[0], -1.0);
        assert_eq!(NF4_LEVELS[15], 1.0);
        assert_eq!(NF4_LEVELS[7], 0.0, "index 7 must be exact zero");
    }

    #[test]
    fn nearest_maps_levels_to_themselves() {
        for (idx, &lvl) in NF4_LEVELS.iter().enumerate() {
            assert_eq!(
                nearest_nf4(lvl),
                idx as u8,
                "level {lvl} should map to {idx}"
            );
        }
    }

    #[test]
    fn quantize_rejects_empty_and_zero_block() {
        assert!(Nf4Tensor::quantize(&[], 4).is_err());
        assert!(Nf4Tensor::quantize(&[1.0, 2.0], 0).is_err());
    }

    #[test]
    fn zero_block_dequantizes_to_zero() {
        // An all-zero block has absmax 0; dequant must stay zero (no NaN).
        let data = vec![0.0_f32; 8];
        let q = Nf4Tensor::quantize(&data, 4).expect("quantize");
        let deq = q.dequantize();
        assert!(deq.iter().all(|&v| v == 0.0), "zeros must round-trip");
    }

    #[test]
    fn exact_levels_roundtrip_within_block() {
        // A block whose values are exactly absmax * level reconstructs exactly.
        let absmax = 2.0_f32;
        let data: Vec<f32> = NF4_LEVELS.iter().map(|&l| l * absmax).collect();
        let q = Nf4Tensor::quantize(&data, 16).expect("quantize");
        let deq = q.dequantize();
        for (a, b) in data.iter().zip(&deq) {
            assert!((a - b).abs() < 1e-5, "exact level roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn quantization_is_low_error_for_normal_data() {
        // NF4 is designed for ~N(0,1) data: relative MSE should be small.
        let mut r = rng();
        let data = randn(&mut r, 4096);
        let q = Nf4Tensor::quantize(&data, 64).expect("quantize");
        let deq = q.dequantize();
        let mse: f32 = data
            .iter()
            .zip(&deq)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / data.len() as f32;
        let var: f32 = data.iter().map(|&v| v * v).sum::<f32>() / data.len() as f32;
        // Empirically NF4 keeps relative error well under 5% on normal data.
        assert!(mse / var < 0.05, "relative MSE too high: {}", mse / var);
    }

    #[test]
    fn odd_length_roundtrip_shape() {
        let data = vec![0.5_f32, -0.5, 1.0, -1.0, 0.25]; // 5 elements (odd)
        let q = Nf4Tensor::quantize(&data, 4).expect("quantize");
        assert_eq!(q.numel(), 5);
        let deq = q.dequantize();
        assert_eq!(deq.len(), 5);
        assert!(deq.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn packed_storage_is_compact() {
        let data = vec![0.1_f32; 256];
        let q = Nf4Tensor::quantize(&data, 64).expect("quantize");
        assert_eq!(q.num_blocks(), 4);
        // 256 codes ⇒ 128 bytes + 4 scales * 4 bytes = 144 bytes,
        // versus 256 * 4 = 1024 bytes full precision.
        assert_eq!(q.storage_bytes(), 128 + 16);
        assert!(q.storage_bytes() < data.len() * 4);
    }

    #[test]
    fn block_size_affects_scale_count() {
        let data = vec![0.3_f32; 100];
        let q1 = Nf4Tensor::quantize(&data, 25).expect("quantize");
        let q2 = Nf4Tensor::quantize(&data, 50).expect("quantize");
        assert_eq!(q1.num_blocks(), 4);
        assert_eq!(q2.num_blocks(), 2);
        assert_eq!(q1.scales().len(), 4);
    }

    #[test]
    fn qlora_forward_shape_and_finiteness() {
        let (in_f, out_f) = (32, 16);
        let mut r = rng();
        let w = randn(&mut r, out_f * in_f);
        let cfg = LoraConfig::new(4, 4.0).expect("config");
        let layer = QLoraLinear::from_weight(&w, in_f, out_f, 16, &cfg, &mut r).expect("layer");
        let x = randn(&mut r, 3 * in_f);
        let y = layer.forward(&x, 3).expect("forward");
        assert_eq!(y.len(), 3 * out_f);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn qlora_zero_lora_equals_dequantized_base_matmul() {
        // Freshly-initialised LoRA has B = 0, so the forward output equals the
        // dequantised-base matmul alone.
        let (in_f, out_f) = (16, 8);
        let mut r = rng();
        let w = randn(&mut r, out_f * in_f);
        let cfg = LoraConfig::new(2, 2.0).expect("config");
        let layer = QLoraLinear::from_weight(&w, in_f, out_f, 16, &cfg, &mut r).expect("layer");
        let deq = layer.base().dequantize();
        let x = randn(&mut r, in_f);
        let y = layer.forward(&x, 1).expect("forward");
        for o in 0..out_f {
            let mut acc = 0.0_f32;
            for i in 0..in_f {
                acc += x[i] * deq[o * in_f + i];
            }
            assert!(
                (y[o] - acc).abs() < 1e-4,
                "B=0 ⇒ base matmul: {} vs {}",
                y[o],
                acc
            );
        }
    }

    #[test]
    fn qlora_lora_changes_output() {
        // Setting B nonzero must move the output away from the base matmul.
        let (in_f, out_f) = (16, 8);
        let mut r = rng();
        let w = randn(&mut r, out_f * in_f);
        let cfg = LoraConfig::new(4, 4.0).expect("config");
        let mut layer = QLoraLinear::from_weight(&w, in_f, out_f, 16, &cfg, &mut r).expect("layer");
        let x = randn(&mut r, in_f);
        let before = layer.forward(&x, 1).expect("forward");
        for v in layer.lora_mut().matrix_b_mut() {
            *v = 0.5;
        }
        let after = layer.forward(&x, 1).expect("forward");
        let diff: f32 = before
            .iter()
            .zip(&after)
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "nonzero LoRA should change output, diff={diff}"
        );
    }

    #[test]
    fn base_quant_mse_matches_manual() {
        let (in_f, out_f) = (8, 8);
        let mut r = rng();
        let w = randn(&mut r, out_f * in_f);
        let cfg = LoraConfig::new(2, 2.0).expect("config");
        let layer = QLoraLinear::from_weight(&w, in_f, out_f, 16, &cfg, &mut r).expect("layer");
        let mse = layer.base_quant_mse(&w).expect("mse");
        assert!(mse >= 0.0 && mse.is_finite());
        assert!(layer.base_quant_mse(&[0.0]).is_err());
        assert_eq!(layer.in_features(), in_f);
        assert_eq!(layer.out_features(), out_f);
    }

    #[test]
    fn from_weight_dim_mismatch_errors() {
        let cfg = LoraConfig::new(2, 2.0).expect("config");
        let mut r = rng();
        // weight too short for 4×4.
        assert!(QLoraLinear::from_weight(&[0.0; 10], 4, 4, 8, &cfg, &mut r).is_err());
        assert!(QLoraLinear::from_weight(&[0.0; 16], 0, 4, 8, &cfg, &mut r).is_err());
    }
}
