/// Scalar quantizer: 8-bit per dimension.
pub struct Sq8Quantizer {
    mins: Vec<f32>,
    scales: Vec<f32>,
    pub dim: usize,
}

impl Sq8Quantizer {
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

    /// Encode one vector to 8-bit codes.
    #[must_use]
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        v.iter()
            .zip(self.mins.iter().zip(self.scales.iter()))
            .map(|(&x, (&mn, &sc))| {
                let normalized = (x - mn) / sc;
                let clamped = normalized.clamp(0.0, 1.0);
                (clamped * 255.0).round() as u8
            })
            .collect()
    }

    /// Decode 8-bit codes back to approximate f32 values.
    #[must_use]
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        codes
            .iter()
            .zip(self.mins.iter().zip(self.scales.iter()))
            .map(|(&c, (&mn, &sc))| mn + (c as f32 / 255.0) * sc)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_approx() {
        let data = vec![0.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let q = Sq8Quantizer::train(&data, 2, 3);
        let v = vec![0.3_f32, 0.6, 0.9];
        let codes = q.encode(&v);
        let decoded = q.decode(&codes);
        for (a, b) in v.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.01, "a={a} b={b}");
        }
    }
}
