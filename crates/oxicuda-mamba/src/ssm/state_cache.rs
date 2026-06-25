//! Streaming SSM state cache — the recurrent-state analogue of a KV cache.
//!
//! # Motivation
//!
//! Attention models stream long contexts with a *KV cache*: the per-token keys
//! and values are stored so generation never re-reads the whole prefix.  A
//! selective SSM has an even cheaper streaming primitive — the recurrence
//!
//! ```text
//! h_t = Ā_t · h_{t-1} + B̄_t · u_t ,   y_t = Σ_n C_t · h_t
//! ```
//!
//! carries **all** of the past in the fixed-size hidden state `h` of shape
//! `[D × N]` (per batch element).  To resume generation we only need to keep
//! `h`, regardless of how long the processed prefix was — `O(D·N)` memory
//! instead of `O(L·D·N)`.
//!
//! [`SsmStateCache`] holds that state and advances it one chunk at a time.
//! Two correctness properties are guaranteed (and unit-tested):
//!
//! 1. **Streaming == full scan.** Processing a sequence as several consecutive
//!    chunks through one cache yields exactly the same `y` as feeding the whole
//!    sequence to [`crate::ssm::parallel_scan::ssm_state_scan`] at once.
//! 2. **Checkpoint / restore.** [`SsmStateCache::checkpoint`] snapshots the
//!    state to a plain `Vec<f32>`; [`SsmStateCache::restore`] (or
//!    [`SsmStateCache::from_checkpoint`]) resumes an *identical* roll-out.  This
//!    is the long-context inference helper: snapshot at any boundary, resume
//!    later without replaying the prefix.
//!
//! All arithmetic is `f32`, matching the other CPU reference kernels.  The
//! per-step parameters are the already-discretized `(Ā, B̄)` plus the
//! input-dependent `C`, exactly as produced by the selective-scan front-end.

use crate::error::{MambaError, MambaResult};

// ─── SsmStateCache ─────────────────────────────────────────────────────────────

/// Persistent hidden state for streaming selective-SSM inference.
///
/// Stores `h` of shape `[D × N]` (row-major, `h[d * N + n]`) for a single
/// sequence (batch element).  Construct with [`SsmStateCache::new`] (zero
/// state) or [`SsmStateCache::from_checkpoint`] (resume), then advance with
/// [`SsmStateCache::step`] / [`SsmStateCache::advance_chunk`].
#[derive(Debug, Clone, PartialEq)]
pub struct SsmStateCache {
    /// Number of channels `D`.
    d_model: usize,
    /// State order `N`.
    d_state: usize,
    /// Hidden state `[D × N]`, row-major.
    h: Vec<f32>,
}

impl SsmStateCache {
    /// Create a zero-initialised cache for a `[D × N]` SSM state.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidModelDim`] — if `d_model == 0`.
    /// * [`MambaError::InvalidSsmOrder`] — if `d_state == 0`.
    pub fn new(d_model: usize, d_state: usize) -> MambaResult<Self> {
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(0));
        }
        if d_state == 0 {
            return Err(MambaError::InvalidSsmOrder(0));
        }
        Ok(Self {
            d_model,
            d_state,
            h: vec![0.0_f32; d_model * d_state],
        })
    }

    /// Number of channels `D`.
    #[inline]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// State order `N`.
    #[inline]
    pub fn d_state(&self) -> usize {
        self.d_state
    }

    /// Read-only view of the current hidden state `[D × N]`.
    #[inline]
    pub fn state(&self) -> &[f32] {
        &self.h
    }

    /// Reset the hidden state to zero (start a fresh sequence, reusing buffers).
    #[inline]
    pub fn reset(&mut self) {
        self.h.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Snapshot the hidden state into an owned `Vec<f32>` of length `D·N`.
    ///
    /// The snapshot can be persisted and later handed to
    /// [`SsmStateCache::restore`] / [`SsmStateCache::from_checkpoint`] to resume
    /// an identical roll-out.
    #[must_use]
    pub fn checkpoint(&self) -> Vec<f32> {
        self.h.clone()
    }

    /// Overwrite the hidden state from a checkpoint of length `D·N`.
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if `snapshot.len() != D·N`.
    pub fn restore(&mut self, snapshot: &[f32]) -> MambaResult<()> {
        let expected = self.d_model * self.d_state;
        if snapshot.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: snapshot.len(),
            });
        }
        self.h.copy_from_slice(snapshot);
        Ok(())
    }

    /// Build a cache directly from a checkpoint snapshot.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidModelDim`] / [`MambaError::InvalidSsmOrder`] — zero dims.
    /// * [`MambaError::DimensionMismatch`] — if `snapshot.len() != D·N`.
    pub fn from_checkpoint(d_model: usize, d_state: usize, snapshot: &[f32]) -> MambaResult<Self> {
        let mut cache = Self::new(d_model, d_state)?;
        cache.restore(snapshot)?;
        Ok(cache)
    }

    /// Advance the cache by a **single** time step and return `y_t` (length `D`).
    ///
    /// # Arguments (all already-discretized, row-major)
    ///
    /// * `u_t`     — input per channel, length `D`.
    /// * `a_bar`   — discretized decay `Ā[d, n]`, length `D·N`.
    /// * `b_bar`   — discretized input gain `B̄[d, n]`, length `D·N`.
    /// * `c_t`     — output projection `C[d, n]`, length `D·N`.
    ///
    /// # Returns
    ///
    /// Output `y_t[d] = Σ_n C[d, n] · h_t[d, n]`, length `D`.
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if any slice length is wrong.
    pub fn step(
        &mut self,
        u_t: &[f32],
        a_bar: &[f32],
        b_bar: &[f32],
        c_t: &[f32],
    ) -> MambaResult<Vec<f32>> {
        let d = self.d_model;
        let n = self.d_state;
        if u_t.len() != d {
            return Err(MambaError::DimensionMismatch {
                expected: d,
                got: u_t.len(),
            });
        }
        let dn = d * n;
        for got in [a_bar.len(), b_bar.len(), c_t.len()] {
            if got != dn {
                return Err(MambaError::DimensionMismatch { expected: dn, got });
            }
        }

        let mut y = vec![0.0_f32; d];
        for ch in 0..d {
            let u_val = u_t[ch];
            let base = ch * n;
            let mut acc = 0.0_f32;
            for k in 0..n {
                let idx = base + k;
                let h_new = a_bar[idx] * self.h[idx] + b_bar[idx] * u_val;
                self.h[idx] = h_new;
                acc += c_t[idx] * h_new;
            }
            y[ch] = acc;
        }
        Ok(y)
    }

    /// Advance the cache over a chunk of `chunk_len` consecutive time steps,
    /// returning the chunk output `y` (row-major `[chunk_len × D]`).
    ///
    /// # Arguments (row-major)
    ///
    /// * `u`         — `[chunk_len × D]`.
    /// * `a_bar`     — `[chunk_len × D × N]` discretized decays.
    /// * `b_bar`     — `[chunk_len × D × N]` discretized input gains.
    /// * `c`         — `[chunk_len × D × N]` output projections.
    /// * `chunk_len` — number of steps in this chunk (`> 0`).
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidSeqLen`]     — if `chunk_len == 0`.
    /// * [`MambaError::DimensionMismatch`] — if any slice length is inconsistent.
    pub fn advance_chunk(
        &mut self,
        u: &[f32],
        a_bar: &[f32],
        b_bar: &[f32],
        c: &[f32],
        chunk_len: usize,
    ) -> MambaResult<Vec<f32>> {
        if chunk_len == 0 {
            return Err(MambaError::InvalidSeqLen(0));
        }
        let d = self.d_model;
        let n = self.d_state;
        let u_expected = chunk_len * d;
        if u.len() != u_expected {
            return Err(MambaError::DimensionMismatch {
                expected: u_expected,
                got: u.len(),
            });
        }
        let p_expected = chunk_len * d * n;
        for got in [a_bar.len(), b_bar.len(), c.len()] {
            if got != p_expected {
                return Err(MambaError::DimensionMismatch {
                    expected: p_expected,
                    got,
                });
            }
        }

        let mut out = vec![0.0_f32; u_expected];
        for t in 0..chunk_len {
            let u_t = &u[t * d..(t + 1) * d];
            let a_t = &a_bar[t * d * n..(t + 1) * d * n];
            let b_t = &b_bar[t * d * n..(t + 1) * d * n];
            let c_t = &c[t * d * n..(t + 1) * d * n];
            let y_t = self.step(u_t, a_t, b_t, c_t)?;
            out[t * d..(t + 1) * d].copy_from_slice(&y_t);
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    /// Discretized parameters for a `[L × D × N]` problem with stable decays.
    fn make_problem(
        rng: &mut LcgRng,
        l: usize,
        d: usize,
        n: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let u = randn(rng, l * d);
        // a_bar in (0, 1) for stability.
        let a_bar: Vec<f32> = (0..l * d * n)
            .map(|_| rng.next_f32() * 0.9 + 0.05)
            .collect();
        let b_bar = randn(rng, l * d * n);
        let c = randn(rng, l * d * n);
        (u, a_bar, b_bar, c)
    }

    /// Reference: run the whole sequence in one chunk through a fresh cache.
    fn full_reference(
        d: usize,
        n: usize,
        l: usize,
        u: &[f32],
        a_bar: &[f32],
        b_bar: &[f32],
        c: &[f32],
    ) -> Vec<f32> {
        let mut cache = SsmStateCache::new(d, n).expect("cache");
        cache.advance_chunk(u, a_bar, b_bar, c, l).expect("full")
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn new_rejects_zero_dims() {
        assert!(matches!(
            SsmStateCache::new(0, 4),
            Err(MambaError::InvalidModelDim(0))
        ));
        assert!(matches!(
            SsmStateCache::new(4, 0),
            Err(MambaError::InvalidSsmOrder(0))
        ));
    }

    #[test]
    fn new_state_is_zero() {
        let cache = SsmStateCache::new(3, 4).expect("cache");
        assert_eq!(cache.state().len(), 12);
        assert!(cache.state().iter().all(|&v| v == 0.0));
        assert_eq!(cache.d_model(), 3);
        assert_eq!(cache.d_state(), 4);
    }

    // ── Streaming == full scan ────────────────────────────────────────────────

    #[test]
    fn streaming_in_chunks_matches_full_scan() {
        let mut rng = LcgRng::new(42);
        let (d, n, l) = (3_usize, 4_usize, 20_usize);
        let (u, a_bar, b_bar, c) = make_problem(&mut rng, l, d, n);

        let reference = full_reference(d, n, l, &u, &a_bar, &b_bar, &c);

        // Process the same sequence in uneven chunks 7 + 8 + 5.
        let mut cache = SsmStateCache::new(d, n).expect("cache");
        let mut streamed = Vec::with_capacity(l * d);
        let mut start = 0_usize;
        for &len in &[7_usize, 8, 5] {
            let u_c = &u[start * d..(start + len) * d];
            let a_c = &a_bar[start * d * n..(start + len) * d * n];
            let b_c = &b_bar[start * d * n..(start + len) * d * n];
            let c_c = &c[start * d * n..(start + len) * d * n];
            let y_c = cache.advance_chunk(u_c, a_c, b_c, c_c, len).expect("chunk");
            streamed.extend_from_slice(&y_c);
            start += len;
        }

        assert_eq!(streamed.len(), reference.len());
        for (i, (&s, &r)) in streamed.iter().zip(reference.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-5,
                "chunk stream mismatch at {i}: {s} vs {r}"
            );
        }
    }

    #[test]
    fn single_steps_match_full_scan() {
        let mut rng = LcgRng::new(7);
        let (d, n, l) = (2_usize, 3_usize, 12_usize);
        let (u, a_bar, b_bar, c) = make_problem(&mut rng, l, d, n);
        let reference = full_reference(d, n, l, &u, &a_bar, &b_bar, &c);

        let mut cache = SsmStateCache::new(d, n).expect("cache");
        for t in 0..l {
            let y_t = cache
                .step(
                    &u[t * d..(t + 1) * d],
                    &a_bar[t * d * n..(t + 1) * d * n],
                    &b_bar[t * d * n..(t + 1) * d * n],
                    &c[t * d * n..(t + 1) * d * n],
                )
                .expect("step");
            for (ch, &yv) in y_t.iter().enumerate() {
                let r = reference[t * d + ch];
                assert!((yv - r).abs() < 1e-5, "t={t} ch={ch}: {yv} vs {r}");
            }
        }
    }

    // ── Checkpoint / restore ──────────────────────────────────────────────────

    #[test]
    fn checkpoint_restore_resumes_identical_rollout() {
        let mut rng = LcgRng::new(123);
        let (d, n, l) = (3_usize, 4_usize, 24_usize);
        let (u, a_bar, b_bar, c) = make_problem(&mut rng, l, d, n);
        let reference = full_reference(d, n, l, &u, &a_bar, &b_bar, &c);

        let split = 10_usize;
        // Run the first `split` steps, then checkpoint.
        let mut cache = SsmStateCache::new(d, n).expect("cache");
        let _ = cache
            .advance_chunk(
                &u[..split * d],
                &a_bar[..split * d * n],
                &b_bar[..split * d * n],
                &c[..split * d * n],
                split,
            )
            .expect("first half");
        let snapshot = cache.checkpoint();

        // Build a *new* cache from the snapshot and finish the sequence.
        let mut resumed = SsmStateCache::from_checkpoint(d, n, &snapshot).expect("from_checkpoint");
        let rest = l - split;
        let y_rest = resumed
            .advance_chunk(
                &u[split * d..],
                &a_bar[split * d * n..],
                &b_bar[split * d * n..],
                &c[split * d * n..],
                rest,
            )
            .expect("second half");

        // The tail output must match the full reference tail exactly.
        for (i, &yv) in y_rest.iter().enumerate() {
            let r = reference[split * d + i];
            assert!((yv - r).abs() < 1e-5, "resume mismatch at {i}: {yv} vs {r}");
        }
    }

    #[test]
    fn restore_rejects_wrong_length() {
        let mut cache = SsmStateCache::new(2, 3).expect("cache");
        assert!(matches!(
            cache.restore(&[0.0; 5]), // should be 6
            Err(MambaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn checkpoint_roundtrip_equals_clone() {
        let mut rng = LcgRng::new(55);
        let (d, n, l) = (2_usize, 2_usize, 5_usize);
        let (u, a_bar, b_bar, c) = make_problem(&mut rng, l, d, n);
        let mut cache = SsmStateCache::new(d, n).expect("cache");
        cache.advance_chunk(&u, &a_bar, &b_bar, &c, l).expect("run");
        let snap = cache.checkpoint();
        let rebuilt = SsmStateCache::from_checkpoint(d, n, &snap).expect("rebuild");
        assert_eq!(cache, rebuilt, "checkpoint round-trip must preserve state");
    }

    // ── reset ─────────────────────────────────────────────────────────────────

    #[test]
    fn reset_zeroes_state_and_restarts() {
        let mut rng = LcgRng::new(9);
        let (d, n, l) = (2_usize, 3_usize, 6_usize);
        let (u, a_bar, b_bar, c) = make_problem(&mut rng, l, d, n);
        let reference = full_reference(d, n, l, &u, &a_bar, &b_bar, &c);

        let mut cache = SsmStateCache::new(d, n).expect("cache");
        // Run once, reset, run again — second run must equal the reference.
        cache
            .advance_chunk(&u, &a_bar, &b_bar, &c, l)
            .expect("run1");
        cache.reset();
        assert!(cache.state().iter().all(|&v| v == 0.0));
        let y2 = cache
            .advance_chunk(&u, &a_bar, &b_bar, &c, l)
            .expect("run2");
        for (i, (&yv, &r)) in y2.iter().zip(reference.iter()).enumerate() {
            assert!((yv - r).abs() < 1e-5, "post-reset mismatch at {i}");
        }
    }

    // ── Errors ────────────────────────────────────────────────────────────────

    #[test]
    fn advance_chunk_errors() {
        let mut cache = SsmStateCache::new(2, 3).expect("cache");
        assert!(matches!(
            cache.advance_chunk(&[], &[], &[], &[], 0),
            Err(MambaError::InvalidSeqLen(0))
        ));
        // chunk_len=2, d=2, n=3 ⇒ u must be 4 (give 3); params must be 12 each.
        assert!(matches!(
            cache.advance_chunk(&[0.0; 3], &[0.0; 12], &[0.0; 12], &[0.0; 12], 2),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn step_errors() {
        let mut cache = SsmStateCache::new(2, 3).expect("cache");
        // u_t wrong length.
        assert!(matches!(
            cache.step(&[0.0], &[0.0; 6], &[0.0; 6], &[0.0; 6]),
            Err(MambaError::DimensionMismatch { .. })
        ));
        // a_bar wrong length.
        assert!(matches!(
            cache.step(&[0.0; 2], &[0.0; 5], &[0.0; 6], &[0.0; 6]),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }
}
