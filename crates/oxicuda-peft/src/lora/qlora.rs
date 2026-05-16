use crate::handle::LcgRng;
use crate::lora::lora::{LoraConfig, mat_vec_mul};

/// The 16 quantile values of a zero-mean unit-normal distribution, normalised to `[-1, 1]`.
///
/// These are the canonical NF4 (NormalFloat4) table entries from the QLoRA paper.
pub const NF4_TABLE: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.3949468731880188,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Quantize a single `f32` value to a 4-bit NF4 index.
///
/// Normalises `val / absmax` into `[-1, 1]`, then returns the index of the nearest NF4 entry.
#[must_use]
pub fn nf4_quantize(val: f32, absmax: f32) -> u8 {
    if absmax == 0.0 {
        return 7; // NF4_TABLE[7] == 0.0 — zero bucket
    }
    let normalised = (val / absmax).clamp(-1.0, 1.0);
    let mut best_idx = 0u8;
    let mut best_dist = (normalised - NF4_TABLE[0]).abs();
    for (i, &entry) in NF4_TABLE.iter().enumerate().skip(1) {
        let dist = (normalised - entry).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Dequantize a 4-bit NF4 index back to a `f32` value.
#[must_use]
pub fn nf4_dequantize(idx: u8, absmax: f32) -> f32 {
    let table_val = NF4_TABLE[idx as usize];
    table_val * absmax
}

/// Quantize a block of `f32` values using per-block NF4 quantization.
///
/// Returns the packed 4-bit code bytes and the per-block absolute maximum.
/// Output `codes` has length `block.len()` (one byte per element for simplicity — not nibble-packed
/// at this CPU-simulation level; the PTX kernel handles nibble packing on-device).
#[must_use]
pub fn quantize_block(block: &[f32]) -> (Vec<u8>, f32) {
    let absmax = block.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
    let codes: Vec<u8> = block.iter().map(|&v| nf4_quantize(v, absmax)).collect();
    (codes, absmax)
}

/// Dequantize a block of NF4 codes using a shared `absmax`.
#[must_use]
pub fn dequantize_block(codes: &[u8], absmax: f32) -> Vec<f32> {
    codes.iter().map(|&c| nf4_dequantize(c, absmax)).collect()
}

/// A quantised linear layer using NF4 weights combined with a LoRA adapter.
///
/// The base weight `W` is stored in 4-bit NF4 format; the LoRA adapter `A`, `B`
/// remain in full `f32` precision.
#[derive(Debug, Clone)]
pub struct QloraLinear {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// LoRA rank.
    pub rank: usize,
    /// Effective LoRA scale α/r.
    pub scale: f32,
    /// NF4-quantised codes for the base weight matrix (one byte per weight element).
    pub codes: Vec<u8>,
    /// Absolute maximum used during weight quantization (single-block, whole-matrix scale).
    pub absmax: f32,
    /// LoRA factor A, shape `[rank × in_features]`.
    pub a: Vec<f32>,
    /// LoRA factor B, shape `[out_features × rank]`.
    pub b: Vec<f32>,
}

impl QloraLinear {
    /// Construct a `QloraLinear` by quantising a pre-existing weight matrix.
    ///
    /// `w` must have length `in_features * out_features` (row-major `[out × in]`).
    #[must_use]
    pub fn from_weights(
        w: &[f32],
        in_features: usize,
        out_features: usize,
        cfg: &LoraConfig,
        rng: &mut LcgRng,
    ) -> Self {
        let scale = cfg.alpha / cfg.r as f32;
        let (codes, absmax) = quantize_block(w);
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
            codes,
            absmax,
            a,
            b,
        }
    }

    /// Compute the forward pass: dequantize W then apply `(W + scale·B·A)·x`.
    ///
    /// `x` must have length `in_features`. Returns a vector of length `out_features`.
    #[must_use]
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Dequantize W
        let w_f32 = dequantize_block(&self.codes, self.absmax);
        // Base: W · x
        let mut out = mat_vec_mul(&w_f32, x, self.out_features, self.in_features);
        // LoRA: scale · B · (A · x)
        let tmp = mat_vec_mul(&self.a, x, self.rank, self.in_features);
        let delta = mat_vec_mul(&self.b, &tmp, self.out_features, self.rank);
        for (o, d) in out.iter_mut().zip(delta.iter()) {
            *o += self.scale * d;
        }
        out
    }
}
