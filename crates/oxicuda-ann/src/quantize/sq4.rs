/// Scalar quantizer: 4-bit per dimension (two values packed per byte).
/// Low nibble = even index, high nibble = odd index.
pub struct Sq4Quantizer {
    mins: Vec<f32>,
    scales: Vec<f32>,
    pub dim: usize,
}

impl Sq4Quantizer {
    /// Train per-dimension min/max from `n` row-major data vectors.
    #[must_use]
    pub fn train(data: &[f32], n: usize, dim: usize) -> Self {
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];

        for row in data.chunks_exact(dim).take(n) {
            for (d, &v) in row.iter().enumerate() {
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }

        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(mn, mx)| {
                let s = mx - mn;
                if s < f32::EPSILON { 1.0 } else { s }
            })
            .collect();

        Self { mins, scales, dim }
    }

    fn quant_one(x: f32, mn: f32, sc: f32) -> u8 {
        let normalized = (x - mn) / sc;
        (normalized.clamp(0.0, 1.0) * 15.0).round() as u8
    }

    fn dequant_one(c: u8, mn: f32, sc: f32) -> f32 {
        mn + (c as f32 / 15.0) * sc
    }

    /// Encode one vector to packed 4-bit codes. Returns `ceil(dim/2)` bytes.
    #[must_use]
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        let nbytes = self.dim.div_ceil(2);
        let mut out = vec![0u8; nbytes];
        for (d, &x) in v.iter().enumerate() {
            let q = Self::quant_one(x, self.mins[d], self.scales[d]);
            let byte_idx = d / 2;
            if d % 2 == 0 {
                out[byte_idx] = q & 0x0F;
            } else {
                out[byte_idx] |= (q & 0x0F) << 4;
            }
        }
        out
    }

    /// Decode packed 4-bit codes back to approximate f32 values.
    #[must_use]
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            let byte_idx = d / 2;
            let nibble = if d % 2 == 0 {
                codes[byte_idx] & 0x0F
            } else {
                (codes[byte_idx] >> 4) & 0x0F
            };
            out.push(Self::dequant_one(nibble, self.mins[d], self.scales[d]));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_4bit() {
        let data = vec![0.0_f32, 0.0, 1.0, 1.0];
        let q = Sq4Quantizer::train(&data, 2, 2);
        let v = vec![0.5_f32, 0.7];
        let codes = q.encode(&v);
        let dec = q.decode(&codes);
        for (a, b) in v.iter().zip(dec.iter()) {
            assert!((a - b).abs() < 0.1, "a={a} b={b}");
        }
    }
}
