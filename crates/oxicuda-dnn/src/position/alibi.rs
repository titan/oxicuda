//! ALiBi — Attention with Linear Biases (Press et al. 2022), CPU reference.
//!
//! "Train Short, Test Long: Attention with Linear Biases Enables Input Length
//! Extrapolation" (Press, Smith & Lewis, ICLR 2022) replaces positional
//! embeddings with a static, non-learned bias added directly to the
//! pre-softmax attention scores. For a query at position `i` attending to a key
//! at position `j` on head `h`:
//!
//! ```text
//! score'[i, j] = score[i, j] − m_h · (i − j)        (causal: j ≤ i)
//! ```
//!
//! The penalty grows linearly with the query/key distance, biasing each head
//! toward recent tokens. The per-head slope `m_h` follows a geometric sequence
//! anchored at `2^(−8 / n_heads)`:
//!
//! ```text
//! ratio  = 2^(−8 / n_heads)
//! m_h    = ratio^(h + 1)        for h ∈ [0, n_heads).
//! ```
//!
//! Because the bias is fixed and distance-only, a model trained at one context
//! length extrapolates to longer ones at inference. This module builds the
//! causal bias matrix `[n_heads × q_len × k_len]`; non-causal positions
//! (`j > i`) are masked to `−∞`.

use crate::error::{DnnError, DnnResult};

/// Per-head ALiBi slope for head `head` of `n_heads`.
///
/// Returns `2^(−8(head + 1) / n_heads)` — the geometric sequence from the
/// paper, applicable when `n_heads` is a power of two (the common case). For
/// arbitrary head counts the same closed form is used, which keeps slopes in
/// `(0, 1]` and monotonically decreasing in `head`.
///
/// # Errors
/// - [`DnnError::InvalidArgument`] if `n_heads == 0` or `head >= n_heads`.
pub fn alibi_slope(head: usize, n_heads: usize) -> DnnResult<f32> {
    if n_heads == 0 || head >= n_heads {
        return Err(DnnError::InvalidArgument(format!(
            "alibi_slope: head {head} out of range for n_heads {n_heads}"
        )));
    }
    let ratio = 2.0_f32.powf(-8.0 / n_heads as f32);
    Ok(ratio.powi((head + 1) as i32))
}

/// Pre-computed causal ALiBi bias matrices.
#[derive(Debug, Clone)]
pub struct AlibiBias {
    /// Bias tensor, flat `[n_heads × q_len × k_len]` row-major. Entries with
    /// `j > i` (future keys) are `f32::NEG_INFINITY`.
    bias: Vec<f32>,
    /// Number of attention heads.
    n_heads: usize,
    /// Number of query positions.
    q_len: usize,
    /// Number of key positions.
    k_len: usize,
}

impl AlibiBias {
    /// Build the causal ALiBi bias for `n_heads` heads over a `q_len × k_len`
    /// score grid.
    ///
    /// Query `i` attends to key `j`; the bias is `−m_h · (i − j)` for `j ≤ i`
    /// and `−∞` for `j > i` (causal mask).
    ///
    /// # Errors
    /// - [`DnnError::InvalidArgument`] if any of `n_heads`, `q_len`, `k_len`
    ///   is zero.
    pub fn new(n_heads: usize, q_len: usize, k_len: usize) -> DnnResult<Self> {
        if n_heads == 0 || q_len == 0 || k_len == 0 {
            return Err(DnnError::InvalidArgument(format!(
                "AlibiBias: n_heads, q_len, k_len must be > 0, got {n_heads}, {q_len}, {k_len}"
            )));
        }

        let mut bias = vec![0.0_f32; n_heads * q_len * k_len];
        for h in 0..n_heads {
            let slope = alibi_slope(h, n_heads)?;
            let head_base = h * q_len * k_len;
            for i in 0..q_len {
                let row_base = head_base + i * k_len;
                for j in 0..k_len {
                    bias[row_base + j] = if j > i {
                        f32::NEG_INFINITY
                    } else {
                        -slope * (i - j) as f32
                    };
                }
            }
        }

        Ok(Self {
            bias,
            n_heads,
            q_len,
            k_len,
        })
    }

    /// Number of heads.
    #[must_use]
    #[inline]
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Query length.
    #[must_use]
    #[inline]
    pub fn q_len(&self) -> usize {
        self.q_len
    }

    /// Key length.
    #[must_use]
    #[inline]
    pub fn k_len(&self) -> usize {
        self.k_len
    }

    /// Borrow the flat `[n_heads × q_len × k_len]` bias tensor.
    #[must_use]
    #[inline]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Add the per-head bias to a flat `[n_heads × q_len × k_len]` score tensor
    /// in-place. Masked (`−∞`) positions remain `−∞`.
    ///
    /// # Errors
    /// - [`DnnError::InvalidDimension`] if `scores.len()` does not equal
    ///   `n_heads · q_len · k_len`.
    pub fn add_to_scores(&self, scores: &mut [f32]) -> DnnResult<()> {
        if scores.len() != self.bias.len() {
            return Err(DnnError::InvalidDimension(format!(
                "AlibiBias::add_to_scores: expected {} elements, got {}",
                self.bias.len(),
                scores.len()
            )));
        }
        for (s, b) in scores.iter_mut().zip(self.bias.iter()) {
            *s += *b;
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slope_power_of_two_matches_geometric() {
        // For n_heads = 8: ratio = 2^(-1) = 0.5; slopes = 0.5, 0.25, …, 1/256.
        let n = 8;
        let expected = [
            0.5_f32, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        ];
        for (h, &e) in expected.iter().enumerate() {
            let s = alibi_slope(h, n).expect("ok");
            assert!((s - e).abs() < 1e-6, "head {h}: got {s}, expected {e}");
        }
    }

    #[test]
    fn slope_monotonic_decreasing() {
        let n = 16;
        let mut prev = f32::INFINITY;
        for h in 0..n {
            let s = alibi_slope(h, n).expect("ok");
            assert!(s < prev, "slopes must decrease: head {h} = {s}");
            assert!(s > 0.0, "slope must be positive");
            prev = s;
        }
    }

    #[test]
    fn slope_out_of_range_error() {
        assert!(matches!(
            alibi_slope(4, 4),
            Err(DnnError::InvalidArgument(_))
        ));
        assert!(matches!(
            alibi_slope(0, 0),
            Err(DnnError::InvalidArgument(_))
        ));
    }

    #[test]
    fn bias_shape() {
        let a = AlibiBias::new(4, 5, 5).expect("ok");
        assert_eq!(a.bias().len(), 4 * 5 * 5);
        assert_eq!(a.n_heads(), 4);
        assert_eq!(a.q_len(), 5);
        assert_eq!(a.k_len(), 5);
    }

    #[test]
    fn diagonal_is_zero() {
        // i == j ⇒ distance 0 ⇒ bias 0 for every head.
        let a = AlibiBias::new(4, 6, 6).expect("ok");
        let bias = a.bias();
        for h in 0..4 {
            for i in 0..6 {
                let v = bias[h * 6 * 6 + i * 6 + i];
                assert!(v.abs() < 1e-9, "diagonal must be 0, got {v}");
            }
        }
    }

    #[test]
    fn future_keys_masked() {
        // j > i must be -inf (causal).
        let a = AlibiBias::new(2, 4, 4).expect("ok");
        let bias = a.bias();
        for h in 0..2 {
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let v = bias[h * 16 + i * 4 + j];
                    assert!(v == f32::NEG_INFINITY, "future key not masked at {i},{j}");
                }
            }
        }
    }

    #[test]
    fn bias_decreases_with_distance() {
        // For a fixed head, more distant past keys get more negative bias.
        let a = AlibiBias::new(1, 5, 5).expect("ok");
        let bias = a.bias();
        let i = 4;
        // j = 4 (dist 0), 3 (dist 1), …, 0 (dist 4): strictly decreasing.
        let mut prev = f32::INFINITY;
        for j in (0..=i).rev() {
            let v = bias[i * 5 + j];
            assert!(v < prev, "bias must decrease with distance: j={j} v={v}");
            prev = v;
        }
    }

    #[test]
    fn add_to_scores_applies_bias() {
        let a = AlibiBias::new(2, 3, 3).expect("ok");
        let mut scores = vec![1.0_f32; 2 * 3 * 3];
        a.add_to_scores(&mut scores).expect("ok");
        let bias = a.bias();
        for (s, b) in scores.iter().zip(bias.iter()) {
            if b.is_finite() {
                assert!((s - (1.0 + b)).abs() < 1e-6);
            } else {
                assert!(*s == f32::NEG_INFINITY, "masked position must stay -inf");
            }
        }
    }

    #[test]
    fn add_to_scores_dim_mismatch_error() {
        let a = AlibiBias::new(2, 3, 3).expect("ok");
        let mut scores = vec![1.0_f32; 10];
        let r = a.add_to_scores(&mut scores);
        assert!(matches!(r, Err(DnnError::InvalidDimension(_))));
    }

    #[test]
    fn new_zero_dim_error() {
        assert!(matches!(
            AlibiBias::new(0, 3, 3),
            Err(DnnError::InvalidArgument(_))
        ));
        assert!(matches!(
            AlibiBias::new(2, 0, 3),
            Err(DnnError::InvalidArgument(_))
        ));
        assert!(matches!(
            AlibiBias::new(2, 3, 0),
            Err(DnnError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rectangular_q_k_lengths() {
        // q_len != k_len (e.g. cross-block attention) must still build.
        let a = AlibiBias::new(2, 3, 6).expect("ok");
        assert_eq!(a.bias().len(), 2 * 3 * 6);
        // Row i=0 sees only key j=0 (others future): j=0 bias 0, j>=1 -inf.
        let bias = a.bias();
        assert!(bias[0].abs() < 1e-9);
        for item in bias.iter().take(6).skip(1) {
            assert!(*item == f32::NEG_INFINITY);
        }
    }
}
