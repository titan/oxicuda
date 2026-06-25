//! Streaming Conformer self-attention with chunked (block) attention and a
//! left-context cache.
//!
//! Standard (offline) Conformer self-attention lets every query frame attend to
//! every key frame in the utterance, which makes it unusable for low-latency
//! streaming recognition: emitting frame `t` would require the whole future of
//! the sequence. This module implements the **chunk-based / time-restricted**
//! self-attention used by streaming Conformer / Transformer-Transducer acoustic
//! models, where the time axis is partitioned into fixed-size chunks and the
//! attention is restricted to a causal-with-left-context window.
//!
//! ## Masking rule
//!
//! Frames are grouped into chunks of `chunk_size` consecutive timesteps. A query
//! frame in chunk `c` may attend to:
//!
//! * **all** key frames inside its own chunk `c` (full *bidirectional*
//!   intra-chunk attention — every frame of the current chunk is already
//!   buffered before the chunk is emitted, so this adds no extra latency); and
//! * **all** key frames in the previous `left_context_chunks` chunks
//!   (`c-1, …, c-left_context_chunks`).
//!
//! It may **not** attend to any frame of a future chunk (`> c`). Disallowed
//! `(query, key)` pairs receive an additive mask of `-inf` before a
//! numerically-stable softmax, so they contribute exactly zero probability.
//!
//! Because chunks are aligned to chunk boundaries, the union of "own chunk +
//! `left_context_chunks` previous chunks" for any query in chunk `c` is exactly
//! the trailing `(left_context_chunks + 1) * chunk_size` frames ending at the
//! last frame of chunk `c`. This is what makes true incremental streaming
//! (one chunk at a time, carrying a small cache of trailing frames) produce the
//! *same* numerical result as the full masked forward over the whole sequence —
//! the central correctness property exercised by the equivalence test below.
//!
//! ## Convention
//!
//! Tensors are flat `Vec<f32>` in row-major, time-major `[T, D]` layout, where
//! `D == embed_dim`. The learned projections `Wq, Wk, Wv, Wo` are each stored as
//! a `[embed_dim, embed_dim]` flat, **output-major** matrix (row index = output
//! dimension, column index = input dimension). A linear layer therefore computes
//! `y[o] = Σ_i x[i] · W[o * embed_dim + i] + b[o]`, i.e. `y = x · Wᵀ`.
//!
//! ## References
//!
//! * X. Chen, Y. Wu, Z. Wang, S. Liu, J. Li (2021),
//!   "Developing Real-time Streaming Transformer Transducer for Speech
//!   Recognition on Large-Scale Dataset", ICASSP 2021 — introduces the
//!   streaming Transformer-Transducer with chunk-wise (block) attention and a
//!   limited left context for low-latency on-device ASR.
//! * C.-C. Wu, M. Wang, et al. (2020), "Streaming Transformer-based Acoustic
//!   Models Using Self-Attention with Augmented Memory" / time-restricted
//!   self-attention — establishes the chunk-based, left-context-only attention
//!   mask that bounds latency while retaining most of the offline accuracy.
//! * A. Gulati et al. (2020), "Conformer: Convolution-augmented Transformer for
//!   Speech Recognition" — the underlying multi-head self-attention block whose
//!   attention pattern this module restricts.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Dense matrix multiply: `C = A · B` where `A` is `[m, k]`, `B` is `[k, n]`.
///
/// All buffers are flat, row-major. Returns the `[m, n]` product.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            let b_row = &b[p * n..p * n + n];
            let c_row = &mut c[i * n..i * n + n];
            for j in 0..n {
                c_row[j] += a_ip * b_row[j];
            }
        }
    }
    c
}

/// Numerically stable in-place softmax over a contiguous slice.
///
/// Subtracts the maximum before exponentiating, then normalises. A `-inf`
/// entry maps to exactly `0`. Empty slices and an all-`-inf` (zero-sum) row are
/// handled gracefully (the latter is left as all-zeros, never producing NaN).
fn softmax_inplace(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_val.is_finite() {
        // Every entry is -inf (fully masked row); leave as zeros.
        for v in scores.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let mut sum = 0.0_f32;
    for v in scores.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in scores.iter_mut() {
            *v /= sum;
        }
    }
}

/// Apply a linear projection `y = x · Wᵀ + b`.
///
/// `x` is `[rows, embed_dim]`, `w` is `[embed_dim, embed_dim]` output-major,
/// `b` is `[embed_dim]`. Returns `[rows, embed_dim]`.
fn linear_proj(x: &[f32], rows: usize, embed_dim: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * embed_dim];
    for r in 0..rows {
        let x_row = &x[r * embed_dim..r * embed_dim + embed_dim];
        let o_row = &mut out[r * embed_dim..r * embed_dim + embed_dim];
        for (o, o_val) in o_row.iter_mut().enumerate() {
            let w_row = &w[o * embed_dim..o * embed_dim + embed_dim];
            let mut acc = b[o];
            for (i, &xv) in x_row.iter().enumerate() {
                acc += xv * w_row[i];
            }
            *o_val = acc;
        }
    }
    out
}

/// Map a frame index to its chunk index (`frame / chunk_size`).
#[inline]
fn chunk_of(frame: usize, chunk_size: usize) -> usize {
    frame / chunk_size
}

/// Streaming attention mask predicate.
///
/// Returns `true` when a query at *absolute* frame `q_frame` is allowed to
/// attend to a key at *absolute* frame `k_frame`, under the chunk-based
/// causal-with-left-context rule. This single predicate is the **only** source
/// of truth for the mask and is shared by both the full forward and the
/// incremental chunk forward so their numerics agree exactly.
#[inline]
fn attends(q_frame: usize, k_frame: usize, chunk_size: usize, left_context_chunks: usize) -> bool {
    let qc = chunk_of(q_frame, chunk_size);
    let kc = chunk_of(k_frame, chunk_size);
    if kc > qc {
        // Future chunk: never allowed.
        return false;
    }
    // kc <= qc. Allowed window is [qc - left_context_chunks, qc].
    let lowest = qc.saturating_sub(left_context_chunks);
    kc >= lowest
}

/// Xavier-uniform half-width `sqrt(6 / (fan_in + fan_out))`.
#[inline]
fn xavier_limit(fan_in: usize, fan_out: usize) -> f32 {
    (6.0 / (fan_in + fan_out) as f32).sqrt()
}

/// Allocate an `[embed_dim, embed_dim]` projection initialised with
/// Box-Muller normal samples scaled by the Xavier limit.
fn init_proj(embed_dim: usize, rng: &mut LcgRng) -> Vec<f32> {
    let lim = xavier_limit(embed_dim, embed_dim);
    let mut w = vec![0.0_f32; embed_dim * embed_dim];
    rng.fill_normal(&mut w);
    for v in w.iter_mut() {
        // fill_normal gives N(0,1); rescale to the Xavier band.
        *v *= lim;
    }
    w
}

/// Core masked multi-head attention shared by `forward` and `forward_chunk`.
///
/// * `q_frames` — flat `[n_q, embed_dim]`, the query rows (already projected).
/// * `kv_frames` — flat `[n_kv, embed_dim]`, the key/value rows (already
///   projected for both K and V; passed separately below).
/// * `q_abs[i]` / `k_abs[j]` — absolute frame index of query `i` / key `j`,
///   used by [`attends`] to build the mask.
///
/// Returns the attention context `[n_q, embed_dim]` (pre output-projection).
#[allow(clippy::too_many_arguments)]
fn masked_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_kv: usize,
    embed_dim: usize,
    n_heads: usize,
    q_abs: &[usize],
    k_abs: &[usize],
    chunk_size: usize,
    left_context_chunks: usize,
) -> Vec<f32> {
    let head_dim = embed_dim / n_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut ctx = vec![0.0_f32; n_q * embed_dim];

    // Precompute the additive mask once per (query, key) pair; it is identical
    // across all heads. Allowed pairs contribute `0`, disallowed `-inf`.
    let mut mask_add = vec![0.0_f32; n_q * n_kv];
    for (qi, &qf) in q_abs.iter().enumerate() {
        for (ki, &kf) in k_abs.iter().enumerate() {
            if !attends(qf, kf, chunk_size, left_context_chunks) {
                mask_add[qi * n_kv + ki] = f32::NEG_INFINITY;
            }
        }
    }

    // Per-head scaled-dot-product attention via `matmul`.
    let mut q_h = vec![0.0_f32; n_q * head_dim];
    let mut k_h_t = vec![0.0_f32; head_dim * n_kv]; // K transposed: [head_dim, n_kv]
    let mut v_h = vec![0.0_f32; n_kv * head_dim];
    for h in 0..n_heads {
        let h_off = h * head_dim;

        // Gather this head's Q, Kᵀ and V slices into contiguous buffers.
        for qi in 0..n_q {
            let src = &q[qi * embed_dim + h_off..qi * embed_dim + h_off + head_dim];
            q_h[qi * head_dim..qi * head_dim + head_dim].copy_from_slice(src);
        }
        for ki in 0..n_kv {
            let k_src = &k[ki * embed_dim + h_off..ki * embed_dim + h_off + head_dim];
            for (d, &kv) in k_src.iter().enumerate() {
                k_h_t[d * n_kv + ki] = kv;
            }
            let v_src = &v[ki * embed_dim + h_off..ki * embed_dim + h_off + head_dim];
            v_h[ki * head_dim..ki * head_dim + head_dim].copy_from_slice(v_src);
        }

        // Scores = (Q_h · K_hᵀ) · scale + mask, then row-softmax.
        let mut scores = matmul(&q_h, &k_h_t, n_q, head_dim, n_kv);
        for (s, m) in scores.iter_mut().zip(mask_add.iter()) {
            *s = *s * scale + *m;
        }
        for qi in 0..n_q {
            softmax_inplace(&mut scores[qi * n_kv..qi * n_kv + n_kv]);
        }

        // Context_h = scores · V_h  →  [n_q, head_dim].
        let ctx_h = matmul(&scores, &v_h, n_q, n_kv, head_dim);
        for qi in 0..n_q {
            ctx[qi * embed_dim + h_off..qi * embed_dim + h_off + head_dim]
                .copy_from_slice(&ctx_h[qi * head_dim..qi * head_dim + head_dim]);
        }
    }
    ctx
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for a [`StreamingConformerAttention`] layer.
#[derive(Debug, Clone)]
pub struct StreamingConformerConfig {
    /// Embedding (model) dimension `D`.
    pub embed_dim: usize,
    /// Number of attention heads (`embed_dim` must be divisible by this).
    pub n_heads: usize,
    /// Number of frames per attention chunk (block).
    pub chunk_size: usize,
    /// Number of *previous* chunks each chunk may additionally attend to.
    ///
    /// A value of `0` makes attention strictly intra-chunk (no left context);
    /// larger values widen the streaming receptive field at the cost of memory.
    pub left_context_chunks: usize,
}

impl StreamingConformerConfig {
    /// Construct and validate a configuration.
    ///
    /// # Errors
    ///
    /// * [`AudioError::InvalidEmbedDim`] when `embed_dim == 0`.
    /// * [`AudioError::InvalidNumHeads`] when `n_heads == 0`.
    /// * [`AudioError::HeadDimMismatch`] when `embed_dim % n_heads != 0`.
    /// * [`AudioError::InvalidSequenceLength`] when `chunk_size == 0`.
    pub fn new(
        embed_dim: usize,
        n_heads: usize,
        chunk_size: usize,
        left_context_chunks: usize,
    ) -> AudioResult<Self> {
        if embed_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if n_heads == 0 {
            return Err(AudioError::InvalidNumHeads(0));
        }
        if embed_dim % n_heads != 0 {
            return Err(AudioError::HeadDimMismatch { embed_dim, n_heads });
        }
        if chunk_size == 0 {
            return Err(AudioError::InvalidSequenceLength(0));
        }
        Ok(Self {
            embed_dim,
            n_heads,
            chunk_size,
            left_context_chunks,
        })
    }

    /// Tiny preset for tests: `D=32, H=4, chunk_size=4, left_context_chunks=2`.
    #[must_use]
    pub fn tiny() -> Self {
        // These values satisfy all `new` invariants by construction.
        Self {
            embed_dim: 32,
            n_heads: 4,
            chunk_size: 4,
            left_context_chunks: 2,
        }
    }

    /// Head dimension `embed_dim / n_heads`.
    #[must_use]
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.n_heads
    }
}

// ─── Left-context cache ──────────────────────────────────────────────────────

/// Rolling cache of trailing input frames for incremental chunk processing.
///
/// Holds at most `left_context_chunks * chunk_size` of the most recent input
/// frames (flat `[n_cached, embed_dim]`, time-major), together with the
/// absolute frame index of the first cached frame so that the streaming mask
/// can be reconstructed exactly. Frames are cached **before** the Q/K/V
/// projections (raw layer input), matching the documented cache design.
#[derive(Debug, Clone)]
pub struct LeftContextCache {
    /// Flat `[n_cached, embed_dim]` trailing input frames.
    frames: Vec<f32>,
    /// Absolute index of the next frame to be consumed (== frames already seen).
    next_frame: usize,
    /// Embedding dimension (for shape checks).
    embed_dim: usize,
    /// Maximum cached frames (`left_context_chunks * chunk_size`).
    capacity: usize,
}

impl LeftContextCache {
    /// Create an empty cache for the given layer geometry.
    #[must_use]
    pub fn new(embed_dim: usize, chunk_size: usize, left_context_chunks: usize) -> Self {
        Self {
            frames: Vec::new(),
            next_frame: 0,
            embed_dim,
            capacity: left_context_chunks * chunk_size,
        }
    }

    /// Number of frames currently cached.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len().checked_div(self.embed_dim).unwrap_or(0)
    }

    /// Whether the cache holds no frames.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Absolute index of the next (not-yet-processed) frame.
    #[must_use]
    #[inline]
    pub fn next_frame(&self) -> usize {
        self.next_frame
    }

    /// Reset the cache to its initial empty state.
    pub fn reset(&mut self) {
        self.frames.clear();
        self.next_frame = 0;
    }
}

// ─── Streaming attention layer ───────────────────────────────────────────────

/// Single streaming Conformer multi-head self-attention layer with chunked
/// (block) attention and a left-context cache.
///
/// Implements scaled-dot-product multi-head self-attention restricted by the
/// chunk-based causal-with-left-context mask described at the module level.
/// Provides both a full masked [`forward`](Self::forward) over an entire
/// sequence and an incremental [`forward_chunk`](Self::forward_chunk) that
/// processes one chunk at a time using a [`LeftContextCache`]; the two are
/// numerically equivalent.
#[derive(Debug, Clone)]
pub struct StreamingConformerAttention {
    /// Query projection `[embed_dim, embed_dim]`, output-major.
    wq: Vec<f32>,
    /// Key projection `[embed_dim, embed_dim]`, output-major.
    wk: Vec<f32>,
    /// Value projection `[embed_dim, embed_dim]`, output-major.
    wv: Vec<f32>,
    /// Output projection `[embed_dim, embed_dim]`, output-major.
    wo: Vec<f32>,
    /// Query bias `[embed_dim]`.
    bq: Vec<f32>,
    /// Key bias `[embed_dim]`.
    bk: Vec<f32>,
    /// Value bias `[embed_dim]`.
    bv: Vec<f32>,
    /// Output bias `[embed_dim]`.
    bo: Vec<f32>,
    /// Layer configuration.
    config: StreamingConformerConfig,
}

impl StreamingConformerAttention {
    /// Build a layer from `config`, initialising all projections with
    /// Xavier-scaled Box-Muller samples drawn from `rng`.
    ///
    /// # Errors
    ///
    /// Never fails for a valid `config`; the signature returns
    /// [`AudioResult`] for API consistency and future-proofing.
    pub fn new(config: StreamingConformerConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        let d = config.embed_dim;
        let wq = init_proj(d, rng);
        let wk = init_proj(d, rng);
        let wv = init_proj(d, rng);
        let wo = init_proj(d, rng);
        // Biases initialised to zero (standard attention init).
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            bq: vec![0.0_f32; d],
            bk: vec![0.0_f32; d],
            bv: vec![0.0_f32; d],
            bo: vec![0.0_f32; d],
            config,
        })
    }

    /// Borrow the layer configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &StreamingConformerConfig {
        &self.config
    }

    /// Create a fresh [`LeftContextCache`] sized for this layer.
    #[must_use]
    pub fn empty_cache(&self) -> LeftContextCache {
        LeftContextCache::new(
            self.config.embed_dim,
            self.config.chunk_size,
            self.config.left_context_chunks,
        )
    }

    /// Apply the output projection `ctx · Woᵀ + bo` to a `[rows, D]` context.
    fn project_out(&self, ctx: &[f32], rows: usize) -> Vec<f32> {
        linear_proj(ctx, rows, self.config.embed_dim, &self.wo, &self.bo)
    }

    /// Full chunked self-attention over an entire `[T, D]` sequence.
    ///
    /// Every query frame attends under the chunk-based causal-with-left-context
    /// mask (own chunk fully + previous `left_context_chunks` chunks; never a
    /// future chunk). Returns `[T, D]` flat.
    ///
    /// # Errors
    ///
    /// * [`AudioError::EmptyInput`] when `t == 0`.
    /// * [`AudioError::ShapeMismatch`] when `x.len() != t * embed_dim`.
    pub fn forward(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        let d = self.config.embed_dim;
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "StreamingConformerAttention::forward: t == 0".into(),
            });
        }
        if x.len() != t * d {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "StreamingConformerAttention::forward: x.len()={} != t*d={}",
                    x.len(),
                    t * d
                ),
            });
        }

        let q = linear_proj(x, t, d, &self.wq, &self.bq);
        let k = linear_proj(x, t, d, &self.wk, &self.bk);
        let v = linear_proj(x, t, d, &self.wv, &self.bv);

        // Absolute indices are simply 0..t for the full forward.
        let abs: Vec<usize> = (0..t).collect();

        let ctx = masked_attention(
            &q,
            &k,
            &v,
            t,
            t,
            d,
            self.config.n_heads,
            &abs,
            &abs,
            self.config.chunk_size,
            self.config.left_context_chunks,
        );

        Ok(self.project_out(&ctx, t))
    }

    /// Process exactly **one** chunk of new frames incrementally, using and
    /// updating the supplied [`LeftContextCache`].
    ///
    /// `chunk` is flat `[n_new, embed_dim]` with `1 <= n_new <= chunk_size`; the
    /// final (ragged) chunk of a sequence may be shorter than `chunk_size`. The
    /// new frames are appended after the cached trailing frames; keys and values
    /// are computed over `[cached | new]`, queries only over `new`, and the
    /// streaming mask is reconstructed from absolute frame indices so that the
    /// result is identical (within floating-point rounding) to the
    /// corresponding rows of [`forward`](Self::forward).
    ///
    /// After computing the output the cache is updated to hold the trailing
    /// `left_context_chunks * chunk_size` frames of `[cached | new]`, ready for
    /// the next call.
    ///
    /// Returns the new chunk's output `[n_new, embed_dim]`.
    ///
    /// # Errors
    ///
    /// * [`AudioError::EmptyInput`] when `chunk` is empty.
    /// * [`AudioError::InvalidSequenceLength`] when `n_new > chunk_size`
    ///   (a single call must not span more than one chunk).
    /// * [`AudioError::ShapeMismatch`] when `chunk.len()` is not a multiple of
    ///   `embed_dim`, or when the cache geometry disagrees with the layer.
    pub fn forward_chunk(
        &self,
        chunk: &[f32],
        cache: &mut LeftContextCache,
    ) -> AudioResult<Vec<f32>> {
        let d = self.config.embed_dim;
        let cs = self.config.chunk_size;

        if chunk.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "StreamingConformerAttention::forward_chunk: empty chunk".into(),
            });
        }
        if chunk.len() % d != 0 {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "forward_chunk: chunk.len()={} not divisible by embed_dim={}",
                    chunk.len(),
                    d
                ),
            });
        }
        if cache.embed_dim != d {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "forward_chunk: cache embed_dim={} != layer embed_dim={}",
                    cache.embed_dim, d
                ),
            });
        }
        let n_new = chunk.len() / d;
        if n_new > cs {
            return Err(AudioError::InvalidSequenceLength(n_new));
        }

        let n_cached = cache.len();
        let cache_start = cache.next_frame.saturating_sub(n_cached);

        // Build the combined [cached | new] buffer.
        let n_kv = n_cached + n_new;
        let mut combined = Vec::with_capacity(n_kv * d);
        combined.extend_from_slice(&cache.frames);
        combined.extend_from_slice(chunk);

        // Absolute indices. Cached frames occupy [cache_start, next_frame) and
        // new frames [next_frame, next_frame + n_new); the two ranges are
        // contiguous, so the keys span [cache_start, next_frame + n_new).
        let k_abs: Vec<usize> = (cache_start..cache.next_frame + n_new).collect();
        let q_abs: Vec<usize> = (cache.next_frame..cache.next_frame + n_new).collect();

        // Projections: K/V over the whole window, Q only over the new frames.
        let k = linear_proj(&combined, n_kv, d, &self.wk, &self.bk);
        let v = linear_proj(&combined, n_kv, d, &self.wv, &self.bv);
        let q = linear_proj(chunk, n_new, d, &self.wq, &self.bq);

        let ctx = masked_attention(
            &q,
            &k,
            &v,
            n_new,
            n_kv,
            d,
            self.config.n_heads,
            &q_abs,
            &k_abs,
            cs,
            self.config.left_context_chunks,
        );
        let out = self.project_out(&ctx, n_new);

        // Update the cache: keep the trailing `capacity` frames of [cached|new].
        let new_next = cache.next_frame + n_new;
        if cache.capacity == 0 {
            cache.frames.clear();
        } else if n_kv > cache.capacity {
            // Retain only the last `capacity` frames of the combined buffer.
            let drop_frames = n_kv - cache.capacity;
            cache.frames = combined[drop_frames * d..].to_vec();
        } else {
            cache.frames = combined;
        }
        cache.next_frame = new_next;

        Ok(out)
    }

    /// Convenience driver: run the whole `[T, D]` sequence through
    /// [`forward_chunk`](Self::forward_chunk) chunk by chunk and concatenate the
    /// per-chunk outputs into `[T, D]`.
    ///
    /// This is exactly how an online recogniser would consume the stream; the
    /// equivalence test confirms it matches [`forward`](Self::forward).
    ///
    /// # Errors
    ///
    /// Propagates errors from [`forward_chunk`](Self::forward_chunk), plus
    /// [`AudioError::EmptyInput`] / [`AudioError::ShapeMismatch`] for an empty or
    /// mis-sized input.
    pub fn forward_streaming(&self, x: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        let d = self.config.embed_dim;
        let cs = self.config.chunk_size;
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "StreamingConformerAttention::forward_streaming: t == 0".into(),
            });
        }
        if x.len() != t * d {
            return Err(AudioError::ShapeMismatch {
                msg: format!("forward_streaming: x.len()={} != t*d={}", x.len(), t * d),
            });
        }

        let mut cache = self.empty_cache();
        let mut out = vec![0.0_f32; t * d];
        let mut start = 0usize;
        while start < t {
            let end = (start + cs).min(t);
            let chunk = &x[start * d..end * d];
            let chunk_out = self.forward_chunk(chunk, &mut cache)?;
            out[start * d..end * d].copy_from_slice(&chunk_out);
            start = end;
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum absolute element-wise difference between two equal-length slices.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    // ── helper: matmul / softmax sanity ───────────────────────────────────────

    #[test]
    fn matmul_simple() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]].
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let c = matmul(&a, &b, 2, 2, 2);
        assert!((c[0] - 19.0).abs() < 1e-4);
        assert!((c[1] - 22.0).abs() < 1e-4);
        assert!((c[2] - 43.0).abs() < 1e-4);
        assert!((c[3] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn matmul_identity() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let eye = vec![1.0_f32, 0.0, 0.0, 1.0];
        let c = matmul(&a, &eye, 2, 2, 2);
        assert_eq!(c, a);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut s = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax_inplace(&mut s);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn softmax_neg_inf_masked_to_zero() {
        let mut s = vec![f32::NEG_INFINITY, 0.0, 1.0, f32::NEG_INFINITY];
        softmax_inplace(&mut s);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[3], 0.0);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn softmax_all_masked_is_zero_no_nan() {
        let mut s = vec![f32::NEG_INFINITY; 4];
        softmax_inplace(&mut s);
        assert!(s.iter().all(|v| *v == 0.0 && v.is_finite()));
    }

    #[test]
    fn softmax_empty_noop() {
        let mut s: Vec<f32> = vec![];
        softmax_inplace(&mut s);
        assert!(s.is_empty());
    }

    // ── mask predicate ────────────────────────────────────────────────────────

    #[test]
    fn attends_blocks_future_chunk() {
        // chunk_size=4, left_context=1. Query frame 1 (chunk 0).
        assert!(attends(1, 0, 4, 1)); // intra-chunk
        assert!(attends(1, 3, 4, 1)); // intra-chunk (bidirectional: future frame, same chunk)
        assert!(!attends(1, 4, 4, 1)); // chunk 1 is a future chunk -> blocked
        assert!(!attends(1, 7, 4, 1));
    }

    #[test]
    fn attends_left_context_window() {
        // chunk_size=2, left_context=1. Query frame 5 (chunk 2).
        // Allowed chunks: 1 and 2 -> frames 2..=5. Frames 0,1 (chunk 0) blocked.
        assert!(!attends(5, 0, 2, 1));
        assert!(!attends(5, 1, 2, 1));
        assert!(attends(5, 2, 2, 1));
        assert!(attends(5, 3, 2, 1));
        assert!(attends(5, 4, 2, 1));
        assert!(attends(5, 5, 2, 1));
    }

    #[test]
    fn attends_zero_left_context_is_intra_only() {
        // left_context=0: only own chunk.
        assert!(attends(6, 6, 4, 0)); // chunk 1
        assert!(attends(6, 4, 4, 0));
        assert!(attends(6, 7, 4, 0));
        assert!(!attends(6, 3, 4, 0)); // chunk 0 blocked
        assert!(!attends(6, 8, 4, 0)); // chunk 2 (future) blocked
    }

    // ── config validation ─────────────────────────────────────────────────────

    #[test]
    fn config_tiny_is_valid() {
        let cfg = StreamingConformerConfig::tiny();
        // Re-validate via `new` with the same fields.
        let rebuilt = StreamingConformerConfig::new(
            cfg.embed_dim,
            cfg.n_heads,
            cfg.chunk_size,
            cfg.left_context_chunks,
        );
        assert!(rebuilt.is_ok());
        assert_eq!(cfg.head_dim(), cfg.embed_dim / cfg.n_heads);
    }

    #[test]
    fn config_zero_embed_dim_err() {
        let r = StreamingConformerConfig::new(0, 4, 4, 1);
        assert_eq!(r.unwrap_err(), AudioError::InvalidEmbedDim(0));
    }

    #[test]
    fn config_zero_heads_err() {
        let r = StreamingConformerConfig::new(32, 0, 4, 1);
        assert_eq!(r.unwrap_err(), AudioError::InvalidNumHeads(0));
    }

    #[test]
    fn config_bad_head_div_err() {
        let r = StreamingConformerConfig::new(32, 7, 4, 1);
        assert_eq!(
            r.unwrap_err(),
            AudioError::HeadDimMismatch {
                embed_dim: 32,
                n_heads: 7,
            }
        );
    }

    #[test]
    fn config_zero_chunk_err() {
        let r = StreamingConformerConfig::new(32, 4, 0, 1);
        assert_eq!(r.unwrap_err(), AudioError::InvalidSequenceLength(0));
    }

    // ── forward shape / finiteness ────────────────────────────────────────────

    #[test]
    fn forward_output_shape_and_finite() {
        let cfg = StreamingConformerConfig::tiny();
        let mut rng = LcgRng::new(42);
        let layer = StreamingConformerAttention::new(cfg.clone(), &mut rng).expect("new");
        let t = 11usize; // ragged: not a multiple of chunk_size=4
        let mut x = vec![0.0_f32; t * cfg.embed_dim];
        let mut xr = LcgRng::new(7);
        xr.fill_normal(&mut x);
        let out = layer.forward(&x, t).expect("forward");
        assert_eq!(out.len(), t * cfg.embed_dim);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn forward_empty_t_err() {
        let cfg = StreamingConformerConfig::tiny();
        let mut rng = LcgRng::new(1);
        let layer = StreamingConformerAttention::new(cfg, &mut rng).expect("new");
        assert!(layer.forward(&[], 0).is_err());
    }

    #[test]
    fn forward_shape_mismatch_err() {
        let cfg = StreamingConformerConfig::tiny();
        let mut rng = LcgRng::new(1);
        let layer = StreamingConformerAttention::new(cfg.clone(), &mut rng).expect("new");
        let x = vec![0.0_f32; 3 * cfg.embed_dim];
        // claim t=5 but provide 3 frames
        assert!(layer.forward(&x, 5).is_err());
    }

    // ── causality: future cannot change a past query ──────────────────────────

    #[test]
    fn future_chunk_does_not_affect_past_query() {
        let cfg = StreamingConformerConfig::new(32, 4, 4, 1).expect("cfg");
        let mut rng = LcgRng::new(123);
        let layer = StreamingConformerAttention::new(cfg.clone(), &mut rng).expect("new");
        let d = cfg.embed_dim;
        let cs = cfg.chunk_size;
        let t = 16usize; // 4 chunks

        let mut x = vec![0.0_f32; t * d];
        let mut xr = LcgRng::new(555);
        xr.fill_normal(&mut x);

        let base = layer.forward(&x, t).expect("forward base");

        // Perturb ONLY frames in the LAST chunk (chunk 3 -> frames 12..16).
        let mut x2 = x.clone();
        let last_chunk_start = (t / cs - 1) * cs; // = 12
        for f in last_chunk_start..t {
            for di in 0..d {
                x2[f * d + di] += 3.0; // large perturbation
            }
        }
        let perturbed = layer.forward(&x2, t).expect("forward perturbed");

        // With left_context_chunks=1, the earliest query that can "see" the last
        // chunk would be in chunk 3 itself. Queries in chunks 0,1,2 must be
        // unchanged. Chunk 2 (frames 8..12) attends to chunks 1,2 only, never 3.
        let safe_frames = last_chunk_start; // frames 0..12 are all in earlier chunks
        let a = &base[..safe_frames * d];
        let b = &perturbed[..safe_frames * d];
        let diff = max_abs_diff(a, b);
        assert!(
            diff < 1e-6,
            "future chunk leaked into past queries: max diff = {diff}"
        );

        // Sanity: the perturbation DID change the last chunk's own outputs,
        // otherwise the test would be vacuous.
        let tail_diff = max_abs_diff(
            &base[last_chunk_start * d..],
            &perturbed[last_chunk_start * d..],
        );
        assert!(
            tail_diff > 1e-4,
            "perturbation had no effect at all (vacuous test): {tail_diff}"
        );
    }

    // ── streaming equivalence (the key property) ──────────────────────────────

    fn assert_streaming_matches_full(cfg: StreamingConformerConfig, t: usize, seed: u64) {
        let mut rng = LcgRng::new(seed);
        let layer = StreamingConformerAttention::new(cfg.clone(), &mut rng).expect("new");
        let d = cfg.embed_dim;
        let mut x = vec![0.0_f32; t * d];
        let mut xr = LcgRng::new(seed ^ 0x9E37_79B9);
        xr.fill_normal(&mut x);

        let full = layer.forward(&x, t).expect("forward");
        let streamed = layer.forward_streaming(&x, t).expect("forward_streaming");
        assert_eq!(full.len(), streamed.len());
        let diff = max_abs_diff(&full, &streamed);
        assert!(
            diff < 1e-4,
            "streaming != full for t={t}, seed={seed}: max diff = {diff}"
        );
    }

    #[test]
    fn streaming_equivalence_exact_multiple() {
        // t = 12 = 3 * chunk_size(4)
        assert_streaming_matches_full(StreamingConformerConfig::tiny(), 12, 1001);
    }

    #[test]
    fn streaming_equivalence_ragged_last_chunk() {
        // t = 14: chunks of 4,4,4,2 -> ragged final chunk.
        assert_streaming_matches_full(StreamingConformerConfig::tiny(), 14, 2002);
    }

    #[test]
    fn streaming_equivalence_single_chunk() {
        // t smaller than one chunk.
        assert_streaming_matches_full(StreamingConformerConfig::tiny(), 3, 3003);
    }

    #[test]
    fn streaming_equivalence_zero_left_context() {
        // No left context: each chunk is fully independent. Still must match.
        let cfg = StreamingConformerConfig::new(32, 4, 4, 0).expect("cfg");
        assert_streaming_matches_full(cfg, 13, 4004);
    }

    #[test]
    fn streaming_equivalence_wide_left_context() {
        // left_context larger than the number of past chunks present early on.
        let cfg = StreamingConformerConfig::new(16, 2, 3, 5).expect("cfg");
        assert_streaming_matches_full(cfg, 17, 5005);
    }

    // ── forward_chunk direct usage / cache bookkeeping ────────────────────────

    #[test]
    fn forward_chunk_rejects_oversized_chunk() {
        let cfg = StreamingConformerConfig::new(8, 2, 3, 1).expect("cfg");
        let mut rng = LcgRng::new(9);
        let layer = StreamingConformerAttention::new(cfg.clone(), &mut rng).expect("new");
        let mut cache = layer.empty_cache();
        // 4 frames > chunk_size(3)
        let big = vec![0.0_f32; 4 * cfg.embed_dim];
        let r = layer.forward_chunk(&big, &mut cache);
        assert_eq!(r.unwrap_err(), AudioError::InvalidSequenceLength(4));
    }

    #[test]
    fn forward_chunk_rejects_empty() {
        let cfg = StreamingConformerConfig::tiny();
        let mut rng = LcgRng::new(9);
        let layer = StreamingConformerAttention::new(cfg, &mut rng).expect("new");
        let mut cache = layer.empty_cache();
        assert!(layer.forward_chunk(&[], &mut cache).is_err());
    }

    #[test]
    fn cache_capacity_is_bounded() {
        let cfg = StreamingConformerConfig::new(8, 2, 2, 2).expect("cfg");
        let cap = cfg.left_context_chunks * cfg.chunk_size; // 4
        let mut rng = LcgRng::new(11);
        let layer = StreamingConformerAttention::new(cfg.clone(), &mut rng).expect("new");
        let mut cache = layer.empty_cache();
        let d = cfg.embed_dim;
        // Feed several chunks; cache must never exceed capacity.
        for c in 0..6 {
            let mut chunk = vec![0.0_f32; cfg.chunk_size * d];
            let mut cr = LcgRng::new(100 + c as u64);
            cr.fill_normal(&mut chunk);
            layer.forward_chunk(&chunk, &mut cache).expect("chunk");
            assert!(
                cache.len() <= cap,
                "cache overflow: {} > {cap}",
                cache.len()
            );
        }
        assert_eq!(cache.next_frame(), 6 * cfg.chunk_size);
    }

    #[test]
    fn cache_reset_clears_state() {
        let cfg = StreamingConformerConfig::tiny();
        let mut rng = LcgRng::new(11);
        let layer = StreamingConformerAttention::new(cfg.clone(), &mut rng).expect("new");
        let mut cache = layer.empty_cache();
        let chunk = vec![0.1_f32; cfg.chunk_size * cfg.embed_dim];
        layer.forward_chunk(&chunk, &mut cache).expect("chunk");
        assert!(!cache.is_empty());
        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.next_frame(), 0);
        assert_eq!(cache.len(), 0);
    }

    // ── determinism ───────────────────────────────────────────────────────────

    #[test]
    fn determinism_same_seed_same_output() {
        let cfg = StreamingConformerConfig::tiny();
        let mut r1 = LcgRng::new(2024);
        let mut r2 = LcgRng::new(2024);
        let l1 = StreamingConformerAttention::new(cfg.clone(), &mut r1).expect("new");
        let l2 = StreamingConformerAttention::new(cfg.clone(), &mut r2).expect("new");
        let t = 10usize;
        let mut x = vec![0.0_f32; t * cfg.embed_dim];
        let mut xr = LcgRng::new(99);
        xr.fill_normal(&mut x);
        let o1 = l1.forward(&x, t).expect("f1");
        let o2 = l2.forward(&x, t).expect("f2");
        assert_eq!(o1, o2, "same seed must give bit-identical output");
    }

    #[test]
    fn different_seed_different_output() {
        let cfg = StreamingConformerConfig::tiny();
        let mut r1 = LcgRng::new(1);
        let mut r2 = LcgRng::new(2);
        let l1 = StreamingConformerAttention::new(cfg.clone(), &mut r1).expect("new");
        let l2 = StreamingConformerAttention::new(cfg.clone(), &mut r2).expect("new");
        let t = 8usize;
        let x = vec![0.3_f32; t * cfg.embed_dim];
        let o1 = l1.forward(&x, t).expect("f1");
        let o2 = l2.forward(&x, t).expect("f2");
        let diff = max_abs_diff(&o1, &o2);
        assert!(diff > 1e-5, "different seeds gave identical output: {diff}");
    }
}
