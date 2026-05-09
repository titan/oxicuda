use crate::error::{AnnError, AnnResult};

/// Hamming distance between two packed u32 slices (popcount of XOR).
pub fn hamming_u32(a: &[u32], b: &[u32]) -> AnnResult<u32> {
    if a.len() != b.len() {
        return Err(AnnError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum())
}

/// Hamming distance between two f32 slices interpreted as bit vectors.
/// Treats each f32 as its 32-bit IEEE754 representation; only `dim` bits are used.
pub fn hamming_f32_packed(a: &[f32], b: &[f32], dim: usize) -> AnnResult<u32> {
    let words = dim.div_ceil(32);
    if a.len() < words || b.len() < words {
        return Err(AnnError::DimensionMismatch {
            expected: words,
            got: a.len().min(b.len()),
        });
    }
    let full_words = dim / 32;
    let rem_bits = dim % 32;

    let mut dist: u32 = 0;
    for i in 0..full_words {
        let xa = a[i].to_bits();
        let xb = b[i].to_bits();
        dist += (xa ^ xb).count_ones();
    }
    if rem_bits > 0 {
        let mask = (1u32 << rem_bits) - 1;
        let xa = a[full_words].to_bits() & mask;
        let xb = b[full_words].to_bits() & mask;
        dist += (xa ^ xb).count_ones();
    }
    Ok(dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_zero_self() {
        let v = vec![0xDEAD_BEEFu32, 0xCAFE_BABE];
        assert_eq!(hamming_u32(&v, &v).unwrap(), 0);
    }

    #[test]
    fn hamming_known() {
        let a = vec![0b0000_0000u32];
        let b = vec![0b1111_1111u32];
        assert_eq!(hamming_u32(&a, &b).unwrap(), 8);
    }
}
