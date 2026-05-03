//! Chunk-wise scan algorithm for Mamba-2 efficiency.
//!
//! # Theory
//!
//! The full SSD recurrence (length L, state N) can be split into chunks of
//! size Q.  Within each chunk, a dense O(Q²·N) intra-chunk product is
//! computed; between chunks, a single O(N) state propagation step carries
//! information forward.  The net complexity is O(L/Q · Q² · N) = O(L·Q·N),
//! which is subquadratic in L when Q ≪ L and matches the recurrent form as
//! Q → 1.
//!
//! The algorithm for chunk `c` with range `[t_start, t_end)`:
//! 1. Compute intra-chunk contribution via [`ssd_naive`] on the chunk.
//! 2. Propagate the inter-chunk contribution from the running state `h_c ∈ Rᴺ`
//!    carried in from previous chunks:
//!    ```text
//!    inter[t] = C[t] · h_c * (Π_{k=t_start}^{t} A[k])
//!    ```
//! 3. Add the two contributions to get the final output in this chunk.
//!
//! The running state is updated at the end of each chunk via the standard
//! recurrence applied to the whole chunk.

use crate::error::{MambaError, MambaResult};
use crate::mamba2::ssd::{ssd_naive, ssd_recurrent};

// ─── ChunkConfig ─────────────────────────────────────────────────────────────

/// Configuration for a chunk-wise SSM scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkConfig {
    /// Total sequence length `L`.
    pub seq_len: usize,
    /// Chunk size `Q`.  The last chunk may be smaller if `Q ∤ L`.
    pub chunk_size: usize,
    /// SSM state dimension `N`.
    pub state_dim: usize,
}

impl ChunkConfig {
    /// Create a validated `ChunkConfig`.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidChunkSize`]  — if `chunk_size == 0`.
    /// * [`MambaError::InvalidSeqLen`]     — if `seq_len == 0`.
    /// * [`MambaError::InvalidSsmOrder`]   — if `state_dim == 0`.
    /// * [`MambaError::Internal`]          — if `chunk_size > seq_len`.
    pub fn new(seq_len: usize, chunk_size: usize, state_dim: usize) -> MambaResult<Self> {
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(seq_len));
        }
        if chunk_size == 0 {
            return Err(MambaError::InvalidChunkSize(chunk_size));
        }
        if state_dim == 0 {
            return Err(MambaError::InvalidSsmOrder(state_dim));
        }
        if chunk_size > seq_len {
            return Err(MambaError::Internal(format!(
                "chunk_size {chunk_size} exceeds seq_len {seq_len}"
            )));
        }
        Ok(Self {
            seq_len,
            chunk_size,
            state_dim,
        })
    }

    /// Number of chunks: `⌈seq_len / chunk_size⌉`.
    #[inline]
    pub fn n_chunks(&self) -> usize {
        self.seq_len.div_ceil(self.chunk_size)
    }

    /// Return the `[start, end)` half-open range of timestep indices for chunk `c`.
    ///
    /// The last chunk is clamped to `seq_len` if `L` is not divisible by `Q`.
    #[inline]
    pub fn chunk_range(&self, chunk_idx: usize) -> (usize, usize) {
        let start = chunk_idx * self.chunk_size;
        let end = (start + self.chunk_size).min(self.seq_len);
        (start, end)
    }
}

// ─── chunk_scan ──────────────────────────────────────────────────────────────

/// Chunk-wise SSM scan that exactly matches [`ssd_recurrent`] output.
///
/// Splits the sequence into chunks of `config.chunk_size` and computes
/// the combined intra-chunk (SSD matrix) and inter-chunk (state propagation)
/// contributions, yielding the same result as the full recurrence but
/// structured for efficient hardware utilisation.
///
/// # Arguments
///
/// * `a_seq` — Per-timestep decay scalars `[L]`.
/// * `b_seq` — B vectors `[L × N]`, row-major.
/// * `c_seq` — C vectors `[L × N]`, row-major.
/// * `x`     — Scalar input per timestep `[L]`.
/// * `config` — Validated chunk configuration.
///
/// # Errors
///
/// * [`MambaError::DimensionMismatch`] — if any slice length mismatches.
/// * Propagates errors from internal SSD computations.
pub fn chunk_scan(
    a_seq: &[f32],
    b_seq: &[f32],
    c_seq: &[f32],
    x: &[f32],
    config: &ChunkConfig,
) -> MambaResult<Vec<f32>> {
    let l = config.seq_len;
    let n = config.state_dim;

    // Validate outer dimensions
    if a_seq.len() != l {
        return Err(MambaError::DimensionMismatch {
            expected: l,
            got: a_seq.len(),
        });
    }
    if b_seq.len() != l * n {
        return Err(MambaError::DimensionMismatch {
            expected: l * n,
            got: b_seq.len(),
        });
    }
    if c_seq.len() != l * n {
        return Err(MambaError::DimensionMismatch {
            expected: l * n,
            got: c_seq.len(),
        });
    }
    if x.len() != l {
        return Err(MambaError::DimensionMismatch {
            expected: l,
            got: x.len(),
        });
    }

    let mut y = vec![0.0_f32; l];
    // h_prev: the hidden state at the boundary entering the current chunk.
    // h[-1] = 0
    let mut h_prev = vec![0.0_f32; n];

    for chunk_idx in 0..config.n_chunks() {
        let (t_start, t_end) = config.chunk_range(chunk_idx);
        let q = t_end - t_start; // actual chunk length (may be < chunk_size for last chunk)

        // ── Slice chunk data ──────────────────────────────────────────────────
        let a_chunk = &a_seq[t_start..t_end];
        let b_chunk = &b_seq[t_start * n..t_end * n];
        let c_chunk = &c_seq[t_start * n..t_end * n];
        let x_chunk = &x[t_start..t_end];

        // ── 1. Intra-chunk contribution via ssd_naive ─────────────────────────
        // y_intra[r] = Σ_{s ≤ r, within chunk} C[t_start+r] · (Π A) · B[t_start+s] · x[t_start+s]
        let y_intra = ssd_naive(a_chunk, b_chunk, c_chunk, x_chunk, q, n)?;

        // ── 2. Inter-chunk contribution from h_prev ───────────────────────────
        // For each position r within the chunk:
        //   inter[r] = C[t_start+r] · h_prev_propagated_to_r
        // where h_prev_propagated_to_r = (Π_{k=t_start}^{t_start+r} A[k]) * h_prev.
        //
        // This is: (A[t_start]*A[t_start+1]*...*A[t_start+r]) * h_prev,
        // then dot with C[t_start+r].
        //
        // Note: "propagated" here means the state enters the chunk and gets
        // multiplied by the decay at each step before any new input is added.
        // The intra-chunk already handles the B-driven inputs; the inter-chunk
        // contribution is purely from the state that crossed the chunk boundary.
        let mut cumulative_decay = 1.0_f32;
        for r in 0..q {
            // Accumulate decay for step (t_start + r): include A[t_start+r].
            cumulative_decay *= a_chunk[r];
            let c_row = &c_chunk[r * n..(r + 1) * n];
            // dot(C[t_start+r], cumulative_decay * h_prev)
            let mut inter_r = 0.0_f32;
            for k in 0..n {
                inter_r += c_row[k] * cumulative_decay * h_prev[k];
            }
            y[t_start + r] = y_intra[r] + inter_r;
        }

        // ── 3. Update h_prev for the next chunk via recurrence ────────────────
        // We need h at the end of this chunk, which is the state after processing
        // all q steps starting from h_prev.  Propagate explicitly:
        //
        // h_new = A[t_start+q-1] * (... (A[t_start] * h_prev + B[t_start]*x[t_start]) ...) + B[t_start+q-1]*x[t_start+q-1]
        //
        // This is exactly the recurrence, so run it by hand (avoids allocating
        // y again just to get the terminal state).
        let mut h_cur = h_prev.clone();
        for r in 0..q {
            let a_r = a_chunk[r];
            let x_r = x_chunk[r];
            let b_row = &b_chunk[r * n..(r + 1) * n];
            for k in 0..n {
                h_cur[k] = a_r * h_cur[k] + b_row[k] * x_r;
            }
        }
        h_prev = h_cur;
    }

    Ok(y)
}

// ─── verify_chunk_equivalence ────────────────────────────────────────────────

/// Verify that [`chunk_scan`] and [`ssd_recurrent`] produce identical output.
///
/// Returns `Ok(true)` if every element satisfies `|chunk[t] - recurrent[t]| ≤ tol`.
///
/// # Errors
///
/// Propagates any error from [`chunk_scan`] or [`ssd_recurrent`].
pub fn verify_chunk_equivalence(
    a_seq: &[f32],
    b_seq: &[f32],
    c_seq: &[f32],
    x: &[f32],
    config: &ChunkConfig,
    tol: f32,
) -> MambaResult<bool> {
    let chunk_out = chunk_scan(a_seq, b_seq, c_seq, x, config)?;
    let recurrent_out = ssd_recurrent(a_seq, b_seq, c_seq, x, config.seq_len, config.state_dim)?;

    let agrees = chunk_out
        .iter()
        .zip(recurrent_out.iter())
        .all(|(&c, &r)| (c - r).abs() <= tol);
    Ok(agrees)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::mamba2::ssd::ssd_recurrent;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_stable_inputs(
        rng: &mut LcgRng,
        seq_len: usize,
        state_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..seq_len).map(|_| 0.1 + rng.next_f32() * 0.8).collect();
        let mut b = vec![0.0_f32; seq_len * state_dim];
        let mut c = vec![0.0_f32; seq_len * state_dim];
        let mut x = vec![0.0_f32; seq_len];
        rng.fill_normal(&mut b);
        rng.fill_normal(&mut c);
        rng.fill_normal(&mut x);
        (a, b, c, x)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ChunkConfig tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Valid configuration is accepted.
    #[test]
    fn chunk_config_valid() {
        let cfg = ChunkConfig::new(16, 4, 2).expect("valid config");
        assert_eq!(cfg.seq_len, 16);
        assert_eq!(cfg.chunk_size, 4);
        assert_eq!(cfg.state_dim, 2);
    }

    /// Zero chunk size must fail.
    #[test]
    fn chunk_config_zero_chunk_size() {
        let err = ChunkConfig::new(8, 0, 2).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidChunkSize(0)));
    }

    /// chunk_size > seq_len must fail.
    #[test]
    fn chunk_config_chunk_exceeds_seq() {
        let err = ChunkConfig::new(4, 8, 2).expect_err("should fail on chunk > seq");
        assert!(matches!(err, MambaError::Internal(_)));
    }

    /// Exact division: seq=8, chunk=4 → 2 chunks.
    #[test]
    fn chunk_config_n_chunks_exact() {
        let cfg = ChunkConfig::new(8, 4, 1).expect("valid");
        assert_eq!(cfg.n_chunks(), 2);
    }

    /// Non-exact: seq=9, chunk=4 → 3 chunks (ceil division).
    #[test]
    fn chunk_config_n_chunks_ceil() {
        let cfg = ChunkConfig::new(9, 4, 1).expect("valid");
        assert_eq!(cfg.n_chunks(), 3);
    }

    /// seq=1, chunk=1 → 1 chunk.
    #[test]
    fn chunk_config_n_chunks_one() {
        let cfg = ChunkConfig::new(1, 1, 1).expect("valid");
        assert_eq!(cfg.n_chunks(), 1);
    }

    /// chunk_range for exact division covers correct indices.
    #[test]
    fn chunk_config_range_exact() {
        let cfg = ChunkConfig::new(8, 4, 1).expect("valid");
        assert_eq!(cfg.chunk_range(0), (0, 4));
        assert_eq!(cfg.chunk_range(1), (4, 8));
    }

    /// chunk_range last chunk is clamped to seq_len.
    #[test]
    fn chunk_config_range_last_clamped() {
        let cfg = ChunkConfig::new(9, 4, 1).expect("valid");
        assert_eq!(cfg.chunk_range(0), (0, 4));
        assert_eq!(cfg.chunk_range(1), (4, 8));
        assert_eq!(cfg.chunk_range(2), (8, 9)); // last chunk has only 1 element
    }

    // ─────────────────────────────────────────────────────────────────────────
    // chunk_scan tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Output length equals seq_len.
    #[test]
    fn chunk_scan_output_shape() {
        let mut rng = LcgRng::new(10);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 12, 3);
        let cfg = ChunkConfig::new(12, 4, 3).expect("valid");
        let y = chunk_scan(&a, &b, &c, &x, &cfg).expect("chunk_scan_output_shape");
        assert_eq!(y.len(), 12);
    }

    /// All outputs are finite.
    #[test]
    fn chunk_scan_output_finite() {
        let mut rng = LcgRng::new(20);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 16, 2);
        let cfg = ChunkConfig::new(16, 4, 2).expect("valid");
        let y = chunk_scan(&a, &b, &c, &x, &cfg).expect("chunk_scan_output_finite");
        for (t, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{t}]={v} not finite");
        }
    }

    /// When chunk_size == seq_len (single chunk), output equals ssd_recurrent.
    #[test]
    fn chunk_scan_single_chunk_equals_ssd() {
        let mut rng = LcgRng::new(30);
        let seq_len = 8_usize;
        let state_dim = 2_usize;
        let (a, b, c, x) = make_stable_inputs(&mut rng, seq_len, state_dim);
        let cfg = ChunkConfig::new(seq_len, seq_len, state_dim).expect("valid");

        let y_chunk = chunk_scan(&a, &b, &c, &x, &cfg).expect("single chunk");
        let y_rec = ssd_recurrent(&a, &b, &c, &x, seq_len, state_dim).expect("ssd_recurrent");

        for (t, (&yc, &yr)) in y_chunk.iter().zip(y_rec.iter()).enumerate() {
            assert!(
                (yc - yr).abs() < 1e-5,
                "chunk vs recurrent mismatch at t={t}: chunk={yc} recurrent={yr}"
            );
        }
    }

    /// L=8, Q=4, N=1: chunk_scan matches ssd_recurrent.
    #[test]
    fn chunk_equivalence_l8_q4_n1() {
        let mut rng = LcgRng::new(40);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 8, 1);
        let cfg = ChunkConfig::new(8, 4, 1).expect("valid");
        let agrees = verify_chunk_equivalence(&a, &b, &c, &x, &cfg, 1e-5).expect("equiv l8q4n1");
        assert!(
            agrees,
            "chunk_scan and ssd_recurrent disagree for L=8, Q=4, N=1"
        );
    }

    /// L=16, Q=4, N=2: chunk_scan matches ssd_recurrent.
    #[test]
    fn chunk_equivalence_l16_q4_n2() {
        let mut rng = LcgRng::new(50);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 16, 2);
        let cfg = ChunkConfig::new(16, 4, 2).expect("valid");
        let agrees = verify_chunk_equivalence(&a, &b, &c, &x, &cfg, 1e-5).expect("equiv l16q4n2");
        assert!(
            agrees,
            "chunk_scan and ssd_recurrent disagree for L=16, Q=4, N=2"
        );
    }

    /// L=16, Q=7, N=1: non-divisible chunk size.
    #[test]
    fn chunk_equivalence_l16_q7_n1() {
        let mut rng = LcgRng::new(60);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 16, 1);
        let cfg = ChunkConfig::new(16, 7, 1).expect("valid");
        let agrees = verify_chunk_equivalence(&a, &b, &c, &x, &cfg, 2e-5).expect("equiv l16q7n1");
        assert!(
            agrees,
            "chunk_scan and ssd_recurrent disagree for L=16, Q=7, N=1"
        );
    }

    /// L=1, Q=1, N=1: edge case with single element.
    #[test]
    fn chunk_equivalence_l1_q1_n1() {
        let a = vec![0.5_f32];
        let b = vec![1.0_f32];
        let c = vec![0.8_f32];
        let x = vec![2.0_f32];
        let cfg = ChunkConfig::new(1, 1, 1).expect("valid");
        let agrees = verify_chunk_equivalence(&a, &b, &c, &x, &cfg, 1e-6).expect("equiv l1q1n1");
        assert!(
            agrees,
            "chunk_scan and ssd_recurrent disagree for L=1, Q=1, N=1"
        );
    }

    /// L=32, Q=8, N=4: larger test.
    #[test]
    fn chunk_equivalence_l32_q8_n4() {
        let mut rng = LcgRng::new(70);
        let (a, b, c, x) = make_stable_inputs(&mut rng, 32, 4);
        let cfg = ChunkConfig::new(32, 8, 4).expect("valid");
        let agrees = verify_chunk_equivalence(&a, &b, &c, &x, &cfg, 2e-4).expect("equiv l32q8n4");
        assert!(
            agrees,
            "chunk_scan and ssd_recurrent disagree for L=32, Q=8, N=4"
        );
    }

    /// Zero state_dim must fail at config level.
    #[test]
    fn chunk_config_zero_state_dim() {
        let err = ChunkConfig::new(8, 4, 0).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSsmOrder(0)));
    }
}
