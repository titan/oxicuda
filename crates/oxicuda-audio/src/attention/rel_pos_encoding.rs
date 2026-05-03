//! Learned relative-position encoding table.
//!
//! Stores a 1-D table of length `2 * max_len - 1` where entry `i` represents
//! the relative displacement `i - (max_len - 1)`.  Displacements outside the
//! representable range are clamped to the nearest boundary entry.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── RelPosEncoding ──────────────────────────────────────────────────────────

/// Learned relative-position encoding table.
///
/// The table has length `2 * max_len - 1`.  Entry at index
/// `(k as isize - q as isize) + (max_len as isize - 1)` gives the learned
/// additive bias for query position `q` attending to key position `k`.
/// Out-of-range indices are clamped to `[0, 2*max_len-2]`.
#[derive(Debug, Clone)]
pub struct RelPosEncoding {
    /// Learned bias entries. Length = `2 * max_len - 1`.
    pub table: Vec<f32>,
    /// Maximum sequence length the table was built for.
    pub max_len: usize,
}

impl RelPosEncoding {
    /// Create a new `RelPosEncoding` with normally-initialised weights.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidSequenceLength`] when `max_len == 0`.
    pub fn new(max_len: usize, rng: &mut LcgRng) -> AudioResult<Self> {
        if max_len == 0 {
            return Err(AudioError::InvalidSequenceLength(0));
        }
        let table_len = 2 * max_len - 1;
        let mut table = vec![0.0_f32; table_len];
        rng.fill_normal(&mut table);
        // Scale by 1/sqrt(table_len) for stable initialisation.
        let scale = 1.0 / (table_len as f32).sqrt();
        for v in &mut table {
            *v *= scale;
        }
        Ok(Self { table, max_len })
    }

    /// Return the learned bias between query position `q` and key position `k`.
    ///
    /// The displacement index `(k as isize - q as isize) + (max_len - 1)` is
    /// clamped to `[0, 2*max_len-2]` before the table look-up, so this method
    /// never panics.
    #[inline]
    pub fn bias(&self, q: usize, k: usize) -> f32 {
        let max_idx = (2 * self.max_len).saturating_sub(2) as isize;
        let raw = k as isize - q as isize + (self.max_len as isize - 1);
        let clamped = raw.clamp(0, max_idx) as usize;
        self.table[clamped]
    }

    /// Build the full `[Q, K]` relative-position bias matrix.
    ///
    /// The returned `Vec` is row-major: element `[q * k_len + k]` holds the
    /// bias for query `q` attending to key `k`.
    pub fn bias_matrix(&self, q_len: usize, k_len: usize) -> Vec<f32> {
        let mut mat = Vec::with_capacity(q_len * k_len);
        for q in 0..q_len {
            for k in 0..k_len {
                mat.push(self.bias(q, k));
            }
        }
        mat
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn rel_pos_encoding_new_table_len() {
        let mut rng = make_rng();
        let enc = RelPosEncoding::new(16, &mut rng).expect("new ok");
        assert_eq!(enc.table.len(), 2 * 16 - 1);
    }

    #[test]
    fn rel_pos_encoding_zero_q_zero_k() {
        let mut rng = make_rng();
        let enc = RelPosEncoding::new(8, &mut rng).expect("new ok");
        // displacement = 0 - 0 + (max_len - 1) = max_len - 1 = centre element
        let centre = enc.max_len - 1;
        assert_eq!(enc.bias(0, 0), enc.table[centre]);
    }

    #[test]
    fn rel_pos_encoding_bias_clamped() {
        let mut rng = make_rng();
        let enc = RelPosEncoding::new(4, &mut rng).expect("new ok");
        // Large k should clamp to last table entry, not panic.
        let v = enc.bias(0, 1000);
        assert!(v.is_finite());
        assert_eq!(v, enc.table[enc.table.len() - 1]);
    }

    #[test]
    fn rel_pos_bias_matrix_shape() {
        let mut rng = make_rng();
        let enc = RelPosEncoding::new(10, &mut rng).expect("new ok");
        let t = 6_usize;
        let mat = enc.bias_matrix(t, t);
        assert_eq!(mat.len(), t * t);
    }

    #[test]
    fn rel_pos_encoding_max_len_one() {
        let mut rng = make_rng();
        let enc = RelPosEncoding::new(1, &mut rng).expect("new ok");
        assert_eq!(enc.table.len(), 1);
    }

    #[test]
    fn rel_pos_encoding_diagonal_consistent() {
        let mut rng = make_rng();
        let enc = RelPosEncoding::new(8, &mut rng).expect("new ok");
        let k_len = 5_usize;
        let mat = enc.bias_matrix(k_len, k_len);
        for q in 0..k_len {
            assert_eq!(mat[q * k_len + q], enc.bias(q, q));
        }
    }

    #[test]
    fn rel_pos_encoding_zero_max_len_err() {
        let mut rng = make_rng();
        assert!(RelPosEncoding::new(0, &mut rng).is_err());
    }

    #[test]
    fn rel_pos_encoding_all_finite() {
        let mut rng = make_rng();
        let enc = RelPosEncoding::new(32, &mut rng).expect("new ok");
        assert!(enc.table.iter().all(|v| v.is_finite()));
    }
}
