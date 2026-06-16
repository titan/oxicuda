//! RetNet retention (Sun et al. 2023, "Retentive Network").
//!
//! # Background
//!
//! RetNet replaces softmax attention with a **retention** mechanism that admits
//! three mathematically-equivalent computation forms sharing one set of
//! parameters:
//!
//! * **Parallel** — `O(L²·d)`, used for training (full `L×L` decay-masked QKᵀ).
//! * **Recurrent** — `O(L·d²)`, used for autoregressive inference (a running
//!   state `S_t = γ S_{t-1} + kₜᵀ vₜ`).
//! * **Chunkwise** — `O(L·C·d)`, a hybrid: parallel inside chunks of size `C`
//!   and recurrent across chunk boundaries.
//!
//! For a single head with per-step scalar decay `γ ∈ (0, 1)`, the retention of
//! query `qₜ` against the past keys/values is
//!
//! ```text
//! Retention(qₜ) = Σ_{s ≤ t} γ^{t−s} (qₜ · kₛ) vₛ
//! ```
//!
//! which the **recurrent** form computes incrementally as
//!
//! ```text
//! S_t = γ S_{t-1} + kₜᵀ vₜ        (S_t ∈ ℝ^{d_k × d_v})
//! oₜ  = qₜ S_t
//! ```
//!
//! Multi-scale retention (MSR) assigns a **different `γ` per head**, following
//! the paper's schedule `γ_h = 1 − 2^{−5−h}` for head index `h`.  We expose
//! the per-head decays directly so callers can override the schedule.
//!
//! # Layout
//!
//! All tensors are flat row-major.  For a single head with key dim `d_k` and
//! value dim `d_v`:
//! * `q`, `k` — `[L × d_k]`, element `(t, j)` at `t·d_k + j`.
//! * `v`      — `[L × d_v]`, element `(t, j)` at `t·d_v + j`.
//! * output   — `[L × d_v]`.

use crate::error::{MambaError, MambaResult};

// ─── RetentionConfig ─────────────────────────────────────────────────────────

/// Configuration for a single-head retention computation.
#[derive(Debug, Clone)]
pub struct RetentionConfig {
    /// Sequence length `L`.
    pub seq_len: usize,
    /// Query / key dimension `d_k`.
    pub d_k: usize,
    /// Value dimension `d_v`.
    pub d_v: usize,
    /// Per-step decay `γ ∈ (0, 1)`.
    pub gamma: f32,
}

impl RetentionConfig {
    /// Create a new retention config.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidSeqLen`]    — if `seq_len == 0`.
    /// * [`MambaError::InvalidModelDim`]  — if `d_k == 0` or `d_v == 0`.
    /// * [`MambaError::Internal`]         — if `gamma ∉ (0, 1]`.
    pub fn new(seq_len: usize, d_k: usize, d_v: usize, gamma: f32) -> MambaResult<Self> {
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(0));
        }
        if d_k == 0 || d_v == 0 {
            return Err(MambaError::InvalidModelDim(d_k.min(d_v)));
        }
        if !(gamma > 0.0 && gamma <= 1.0) {
            return Err(MambaError::Internal(format!(
                "retention gamma must be in (0, 1], got {gamma}"
            )));
        }
        Ok(Self {
            seq_len,
            d_k,
            d_v,
            gamma,
        })
    }
}

/// Per-head decay schedule from the RetNet paper: `γ_h = 1 − 2^{−5−h}`.
///
/// `n_heads` decays are returned, monotonically increasing toward 1.
#[must_use]
pub fn msr_decays(n_heads: usize) -> Vec<f32> {
    (0..n_heads)
        .map(|h| 1.0_f32 - 2.0_f32.powi(-5 - h as i32))
        .collect()
}

#[inline]
fn check_shapes(cfg: &RetentionConfig, q: &[f32], k: &[f32], v: &[f32]) -> MambaResult<()> {
    let lk = cfg.seq_len * cfg.d_k;
    let lv = cfg.seq_len * cfg.d_v;
    if q.len() != lk {
        return Err(MambaError::DimensionMismatch {
            expected: lk,
            got: q.len(),
        });
    }
    if k.len() != lk {
        return Err(MambaError::DimensionMismatch {
            expected: lk,
            got: k.len(),
        });
    }
    if v.len() != lv {
        return Err(MambaError::DimensionMismatch {
            expected: lv,
            got: v.len(),
        });
    }
    Ok(())
}

// ─── Parallel form ───────────────────────────────────────────────────────────

/// Parallel retention: builds the decay-masked `L×L` score matrix.
///
/// `D[t, s] = γ^{t−s}` for `s ≤ t`, else `0`; output `oₜ = Σ_s D[t,s] (qₜ·kₛ) vₛ`.
///
/// # Errors
///
/// [`MambaError::DimensionMismatch`] on any shape disagreement.
pub fn retention_parallel(
    cfg: &RetentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> MambaResult<Vec<f32>> {
    check_shapes(cfg, q, k, v)?;
    let l = cfg.seq_len;
    let dk = cfg.d_k;
    let dv = cfg.d_v;
    let gamma = cfg.gamma;

    let mut out = vec![0.0_f32; l * dv];
    for t in 0..l {
        let q_t = &q[t * dk..t * dk + dk];
        for s in 0..=t {
            // Inner product qₜ·kₛ.
            let k_s = &k[s * dk..s * dk + dk];
            let mut score = 0.0_f32;
            for j in 0..dk {
                score += q_t[j] * k_s[j];
            }
            // Decay γ^{t−s}.
            let decay = gamma.powi((t - s) as i32);
            let w = score * decay;
            let v_s = &v[s * dv..s * dv + dv];
            let o_t = &mut out[t * dv..t * dv + dv];
            for j in 0..dv {
                o_t[j] += w * v_s[j];
            }
        }
    }
    Ok(out)
}

// ─── Recurrent form ──────────────────────────────────────────────────────────

/// Recurrent retention state `S ∈ ℝ^{d_k × d_v}`.
#[derive(Debug, Clone)]
pub struct RetentionState {
    /// Flattened `[d_k × d_v]` state, `(i, j)` at `i·d_v + j`.
    pub s: Vec<f32>,
    d_k: usize,
    d_v: usize,
}

impl RetentionState {
    /// Allocate a zero state for the given dimensions.
    #[must_use]
    pub fn zeros(d_k: usize, d_v: usize) -> Self {
        Self {
            s: vec![0.0_f32; d_k * d_v],
            d_k,
            d_v,
        }
    }

    /// Advance one step: `S ← γ S + kₜᵀ vₜ`, return `oₜ = qₜ S`.
    ///
    /// `q_t`, `k_t` are length `d_k`; `v_t` is length `d_v`.
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if any input length is wrong.
    pub fn step(
        &mut self,
        q_t: &[f32],
        k_t: &[f32],
        v_t: &[f32],
        gamma: f32,
    ) -> MambaResult<Vec<f32>> {
        let dk = self.d_k;
        let dv = self.d_v;
        if q_t.len() != dk || k_t.len() != dk {
            return Err(MambaError::DimensionMismatch {
                expected: dk,
                got: q_t.len().min(k_t.len()),
            });
        }
        if v_t.len() != dv {
            return Err(MambaError::DimensionMismatch {
                expected: dv,
                got: v_t.len(),
            });
        }
        // S ← γ S + kₜᵀ vₜ
        for (row, &ki) in self.s.chunks_mut(dv).zip(k_t.iter()) {
            for (rj, &vj) in row.iter_mut().zip(v_t.iter()) {
                *rj = gamma * *rj + ki * vj;
            }
        }
        // oₜ = qₜ S  (length d_v)
        let mut o = vec![0.0_f32; dv];
        for (row, &qi) in self.s.chunks(dv).zip(q_t.iter()) {
            for (oj, &rj) in o.iter_mut().zip(row.iter()) {
                *oj += qi * rj;
            }
        }
        Ok(o)
    }
}

/// Recurrent retention over a full sequence (inference form).
///
/// # Errors
///
/// [`MambaError::DimensionMismatch`] on any shape disagreement.
pub fn retention_recurrent(
    cfg: &RetentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> MambaResult<Vec<f32>> {
    check_shapes(cfg, q, k, v)?;
    let l = cfg.seq_len;
    let dk = cfg.d_k;
    let dv = cfg.d_v;
    let mut state = RetentionState::zeros(dk, dv);
    let mut out = vec![0.0_f32; l * dv];
    for t in 0..l {
        let o = state.step(
            &q[t * dk..t * dk + dk],
            &k[t * dk..t * dk + dk],
            &v[t * dv..t * dv + dv],
            cfg.gamma,
        )?;
        out[t * dv..t * dv + dv].copy_from_slice(&o);
    }
    Ok(out)
}

// ─── Chunkwise form ──────────────────────────────────────────────────────────

/// Chunkwise retention: parallel within chunks, recurrent across chunks.
///
/// Splits the sequence into `⌈L / chunk_size⌉` chunks.  Within a chunk the
/// intra-chunk contribution is computed in parallel; the **cross-chunk**
/// contribution `qₜ (γ^{i+1} S_prev)` carries the running state `S` (where `i`
/// is the position inside the chunk).  The two contributions are summed.
///
/// # Errors
///
/// * [`MambaError::InvalidChunkSize`] — if `chunk_size == 0`.
/// * [`MambaError::DimensionMismatch`] — on shape disagreement.
pub fn retention_chunkwise(
    cfg: &RetentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    chunk_size: usize,
) -> MambaResult<Vec<f32>> {
    check_shapes(cfg, q, k, v)?;
    if chunk_size == 0 {
        return Err(MambaError::InvalidChunkSize(0));
    }
    let l = cfg.seq_len;
    let dk = cfg.d_k;
    let dv = cfg.d_v;
    let gamma = cfg.gamma;

    let mut out = vec![0.0_f32; l * dv];
    // Running cross-chunk state S ∈ ℝ^{d_k × d_v} (state *before* the chunk).
    let mut s_state = vec![0.0_f32; dk * dv];

    let n_chunks = l.div_ceil(chunk_size);
    for c in 0..n_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(l);
        let clen = end - start;

        // ── Intra-chunk (parallel) + cross-chunk (state) per position ────────
        for i in 0..clen {
            let t = start + i;
            let q_t = &q[t * dk..t * dk + dk];
            let o_t = &mut out[t * dv..t * dv + dv];

            // Cross-chunk: oₜ += γ^{i+1} · (qₜ S_prev).
            let cross_decay = gamma.powi(i as i32 + 1);
            for ii in 0..dk {
                let qi = q_t[ii] * cross_decay;
                let row = &s_state[ii * dv..ii * dv + dv];
                for j in 0..dv {
                    o_t[j] += qi * row[j];
                }
            }
            // Intra-chunk: causal within chunk.
            for s_idx in 0..=i {
                let s = start + s_idx;
                let k_s = &k[s * dk..s * dk + dk];
                let mut score = 0.0_f32;
                for jj in 0..dk {
                    score += q_t[jj] * k_s[jj];
                }
                let decay = gamma.powi((i - s_idx) as i32);
                let w = score * decay;
                let v_s = &v[s * dv..s * dv + dv];
                for j in 0..dv {
                    o_t[j] += w * v_s[j];
                }
            }
        }

        // ── Update running state with this chunk: S ← γ^{clen} S + Σ_i γ^{clen−1−i} kᵢᵀ vᵢ
        // Decay the carried state by the full chunk length.
        let chunk_decay = gamma.powi(clen as i32);
        for v_ in s_state.iter_mut() {
            *v_ *= chunk_decay;
        }
        for i in 0..clen {
            let t = start + i;
            let k_t = &k[t * dk..t * dk + dk];
            let v_t = &v[t * dv..t * dv + dv];
            let w = gamma.powi((clen - 1 - i) as i32);
            for ii in 0..dk {
                let ki = k_t[ii] * w;
                let row = &mut s_state[ii * dv..ii * dv + dv];
                for j in 0..dv {
                    row[j] += ki * v_t[j];
                }
            }
        }
    }
    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rand_vec(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    fn cfg(l: usize, dk: usize, dv: usize, g: f32) -> RetentionConfig {
        RetentionConfig::new(l, dk, dv, g).expect("cfg")
    }

    #[test]
    fn config_rejects_bad_dims() {
        assert!(matches!(
            RetentionConfig::new(0, 2, 2, 0.9),
            Err(MambaError::InvalidSeqLen(0))
        ));
        assert!(matches!(
            RetentionConfig::new(4, 0, 2, 0.9),
            Err(MambaError::InvalidModelDim(_))
        ));
        assert!(matches!(
            RetentionConfig::new(4, 2, 0, 0.9),
            Err(MambaError::InvalidModelDim(_))
        ));
    }

    #[test]
    fn config_rejects_bad_gamma() {
        assert!(RetentionConfig::new(4, 2, 2, 0.0).is_err());
        assert!(RetentionConfig::new(4, 2, 2, 1.5).is_err());
        assert!(RetentionConfig::new(4, 2, 2, -0.1).is_err());
        assert!(RetentionConfig::new(4, 2, 2, 1.0).is_ok());
    }

    #[test]
    fn msr_decays_schedule() {
        let d = msr_decays(4);
        assert_eq!(d.len(), 4);
        // γ_0 = 1 − 2^{−5} = 0.96875
        assert!((d[0] - 0.96875).abs() < 1e-6);
        // monotone increasing, all < 1
        for w in d.windows(2) {
            assert!(w[1] > w[0]);
        }
        assert!(d.iter().all(|&g| g < 1.0 && g > 0.0));
    }

    #[test]
    fn parallel_shape_and_finite() {
        let mut rng = LcgRng::new(1);
        let c = cfg(6, 3, 4, 0.9);
        let q = rand_vec(&mut rng, 6 * 3);
        let k = rand_vec(&mut rng, 6 * 3);
        let v = rand_vec(&mut rng, 6 * 4);
        let o = retention_parallel(&c, &q, &k, &v).expect("par");
        assert_eq!(o.len(), 6 * 4);
        assert!(o.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn parallel_rejects_bad_shapes() {
        let c = cfg(4, 2, 2, 0.9);
        assert!(retention_parallel(&c, &[0.0; 7], &[0.0; 8], &[0.0; 8]).is_err());
        assert!(retention_parallel(&c, &[0.0; 8], &[0.0; 7], &[0.0; 8]).is_err());
        assert!(retention_parallel(&c, &[0.0; 8], &[0.0; 8], &[0.0; 7]).is_err());
    }

    #[test]
    fn parallel_equals_recurrent() {
        let mut rng = LcgRng::new(7);
        let c = cfg(10, 4, 3, 0.85);
        let q = rand_vec(&mut rng, 10 * 4);
        let k = rand_vec(&mut rng, 10 * 4);
        let v = rand_vec(&mut rng, 10 * 3);
        let par = retention_parallel(&c, &q, &k, &v).expect("par");
        let rec = retention_recurrent(&c, &q, &k, &v).expect("rec");
        for (a, b) in par.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-3, "par {a} vs rec {b}");
        }
    }

    #[test]
    fn parallel_equals_chunkwise() {
        let mut rng = LcgRng::new(13);
        let c = cfg(12, 3, 3, 0.9);
        let q = rand_vec(&mut rng, 12 * 3);
        let k = rand_vec(&mut rng, 12 * 3);
        let v = rand_vec(&mut rng, 12 * 3);
        let par = retention_parallel(&c, &q, &k, &v).expect("par");
        for &cs in &[1usize, 2, 4, 5, 12, 16] {
            let chunk = retention_chunkwise(&c, &q, &k, &v, cs).expect("chunk");
            for (i, (a, b)) in par.iter().zip(chunk.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 2e-3,
                    "chunk_size {cs} idx {i}: par {a} vs chunk {b}"
                );
            }
        }
    }

    #[test]
    fn chunkwise_rejects_zero_chunk() {
        let c = cfg(4, 2, 2, 0.9);
        assert!(matches!(
            retention_chunkwise(&c, &[0.0; 8], &[0.0; 8], &[0.0; 8], 0),
            Err(MambaError::InvalidChunkSize(0))
        ));
    }

    #[test]
    fn recurrent_state_step_shapes() {
        let mut st = RetentionState::zeros(3, 2);
        let o = st
            .step(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[2.0, 3.0], 0.9)
            .expect("step");
        assert_eq!(o.len(), 2);
        // First step: S = kᵀv, o = q S = [2, 3] when q=k=e0.
        assert!((o[0] - 2.0).abs() < 1e-6);
        assert!((o[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn recurrent_state_rejects_bad_len() {
        let mut st = RetentionState::zeros(2, 2);
        assert!(st.step(&[1.0], &[1.0, 0.0], &[1.0, 1.0], 0.9).is_err());
        assert!(st.step(&[1.0, 0.0], &[1.0, 0.0], &[1.0], 0.9).is_err());
    }

    #[test]
    fn decay_reduces_distant_contribution() {
        // A single key/value at t=0 contributes γ^t to later queries.
        let c = cfg(4, 1, 1, 0.5);
        // q = all ones, k has a spike only at t=0, v spike at t=0.
        let q = vec![1.0_f32; 4];
        let mut k = vec![0.0_f32; 4];
        k[0] = 1.0;
        let mut v = vec![0.0_f32; 4];
        v[0] = 1.0;
        let o = retention_parallel(&c, &q, &k, &v).expect("par");
        // o[t] = γ^t · (q·k0) · v0 = 0.5^t.
        assert!((o[0] - 1.0).abs() < 1e-6);
        assert!((o[1] - 0.5).abs() < 1e-6);
        assert!((o[2] - 0.25).abs() < 1e-6);
        assert!((o[3] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn causality_future_does_not_leak() {
        // Output at t=0 must not depend on keys/values at t>0.
        let c = cfg(3, 2, 2, 0.9);
        let q = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v0 = vec![1.0, 1.0, 5.0, 5.0, 9.0, 9.0];
        let mut v1 = v0.clone();
        v1[4] = -100.0; // change a future value
        v1[5] = -100.0;
        let o0 = retention_parallel(&c, &q, &k, &v0).expect("p0");
        let o1 = retention_parallel(&c, &q, &k, &v1).expect("p1");
        // o[0] unaffected by v at t=2.
        assert!((o0[0] - o1[0]).abs() < 1e-9);
        assert!((o0[1] - o1[1]).abs() < 1e-9);
    }

    #[test]
    fn zero_input_zero_output() {
        let c = cfg(5, 2, 3, 0.9);
        let z = vec![0.0_f32; 5 * 2];
        let zv = vec![0.0_f32; 5 * 3];
        let o = retention_parallel(&c, &z, &z, &zv).expect("p");
        assert!(o.iter().all(|&x| x == 0.0));
        let o2 = retention_recurrent(&c, &z, &z, &zv).expect("r");
        assert!(o2.iter().all(|&x| x == 0.0));
    }
}
