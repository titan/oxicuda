use crate::handle::LcgRng;
use crate::lora::lora::{LoraConfig, mat_vec_mul};

/// The 16 quantile values of a zero-mean unit-normal distribution, normalised to `[-1, 1]`.
///
/// These are the canonical NF4 (NormalFloat4) table entries from the QLoRA paper.
pub const NF4_TABLE: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Largest gap between consecutive NF4 codebook entries — the worst-case
    /// nearest-neighbour quantisation interval. Computed from the table itself so
    /// the bound stays correct if the table is ever re-tuned.
    fn max_codebook_gap() -> f32 {
        NF4_TABLE
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn nf4_dequant_bit_exact_to_table() {
        // Anchor points pin the canonical NF4 layout.
        assert_eq!(NF4_TABLE[0], -1.0);
        assert_eq!(NF4_TABLE[7], 0.0);
        assert_eq!(NF4_TABLE[15], 1.0);
        // Strictly ascending.
        for w in NF4_TABLE.windows(2) {
            assert!(
                w[0] < w[1],
                "NF4_TABLE not strictly sorted: {} >= {}",
                w[0],
                w[1]
            );
        }
        // `nf4_dequantize(idx, absmax)` must equal `NF4_TABLE[idx] * absmax` bit-for-
        // bit for every codebook index and every scale (it is a single multiply).
        for &absmax in &[1.0_f32, 3.0, 0.25] {
            for (idx, &entry) in NF4_TABLE.iter().enumerate() {
                let got = nf4_dequantize(idx as u8, absmax);
                assert_eq!(
                    got,
                    entry * absmax,
                    "nf4_dequantize({idx}, {absmax}) not bit-exact to table"
                );
            }
        }
    }

    #[test]
    fn nf4_quantize_dequantize_identity_on_codebook() {
        // A value sitting exactly on a (scaled) codebook point must quantize to that
        // point's index and dequantize back to itself — round-trip identity.
        for &absmax in &[1.0_f32, 2.5] {
            for (idx, &entry) in NF4_TABLE.iter().enumerate() {
                let val = entry * absmax;
                let code = nf4_quantize(val, absmax);
                assert_eq!(
                    code, idx as u8,
                    "codebook point {val} did not quantize to index {idx}"
                );
                assert_eq!(
                    nf4_dequantize(code, absmax),
                    val,
                    "quantize∘dequantize not identity at index {idx}"
                );
            }
        }
    }

    #[test]
    fn nf4_block_roundtrip_exact_on_codebook_points() {
        // A block whose entries are all scaled codebook points round-trips exactly:
        // the block absmax is 2.0 (|±1.0|·2 dominates), every entry normalises back
        // to its table value, re-quantizes to its own index, and dequantizes exactly.
        let block: Vec<f32> = NF4_TABLE.iter().map(|&e| e * 2.0).collect();
        let (codes, absmax) = quantize_block(&block);
        assert_eq!(absmax, 2.0);
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(c, i as u8, "codebook block code mismatch at index {i}");
        }
        let dequant = dequantize_block(&codes, absmax);
        assert_eq!(dequant, block, "codebook block did not round-trip exactly");
    }

    #[test]
    fn nf4_roundtrip_error_bounded_by_codebook_spacing() {
        // For arbitrary values inside the representable range, nearest-neighbour
        // quantisation error is at most half the largest codebook gap, scaled by the
        // block's absmax. Round-trip deterministic pseudo-random values and assert it.
        let max_gap = max_codebook_gap();
        for &amp in &[1.0_f32, 2.0] {
            let mut rng = LcgRng::new(2024);
            let vals: Vec<f32> = (0..512)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * amp)
                .collect();
            let (codes, absmax) = quantize_block(&vals);
            let dequant = dequantize_block(&codes, absmax);
            let max_err = vals
                .iter()
                .zip(dequant.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let bound = 0.5 * max_gap * absmax + 1e-6;
            assert!(
                max_err <= bound,
                "NF4 round-trip error {max_err} exceeds half-gap bound {bound} (absmax={absmax})"
            );
        }
    }
}
