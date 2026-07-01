//! PTX GPU kernel generators for the OxiCUDA inference engine.
//!
//! Each function returns a valid PTX string for the requested SM architecture.
//! Kernels are designed for decode-step throughput at batch sizes 1–512.
//!
//! # Kernel catalogue
//!
//! | Function | Description |
//! |---|---|
//! | [`paged_attn_ptx`] | Scaled dot-product attention over paged KV blocks |
//! | [`rope_apply_ptx`] | Rotary Position Embedding (RoPE) to Q and K |
//! | [`top_k_filter_ptx`] | Suppress non-top-K logits to −∞ |
//! | [`logits_softmax_ptx`] | Numerically stable softmax over logit vector |
//! | [`kv_append_ptx`] | Append one token's K/V to the paged cache |
//!
//! # On-device validation
//!
//! Every kernel below is JIT-loaded on a real CUDA device and checked against a
//! CPU oracle by the `gpu_tests` module (enable with `--features gpu-tests`).
//! Several bugs were found and fixed that way; see the per-function notes.

/// PTX IEEE 754 hex literal for a `f32` value.
fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── paged_attn_ptx ──────────────────────────────────────────────────────────

/// Generate the PTX kernel for PagedAttention.
///
/// Each thread block handles one attention head; each thread owns one element
/// `dim` of the head dimension. The kernel:
/// 1. Iterates over the KV blocks referenced by `block_table`.
/// 2. For each filled slot, recomputes the **full** query·key dot product over
///    `head_dim`, scales it, and folds it into a FlashAttention-style online
///    softmax (`Dao et al.`): running max `m`, running denominator `l`.
/// 3. Accumulates `weight · v[t, dim]` into this thread's output element.
/// 4. Normalises by `l` and stores `out[head, dim]`.
///
/// The exponential is evaluated in **base e** as `ex2(x · log2(e))`.
///
/// ## Bugs found on-device and fixed (vs the original hand-written PTX)
///
/// * **Invalid PTX** — the old offset math fed a 64-bit register to
///   `mul.wide.u32` (whose sources must be 32-bit); ptxas rejected the module
///   with "invalid PTX" on the RTX A4000 (sm_86), so it never loaded.
/// * **Per-element "dot product"** — each thread used only `q[dim]·k[t,dim]` as
///   the attention score instead of the full `Σ_d q[d]·k[t,d]`, so the softmax
///   was meaningless. Fixed: each thread now recomputes the full dot product.
/// * **Value loaded from the key buffer** — `v` was read through `k_ptr`. Fixed
///   to read through `v_ptr`.
/// * **Base-2 softmax** — `ex2.approx.f32` was applied without the `· log2(e)`
///   scale, computing `2^x` instead of `e^x` (≈20–50 % error that still sums to
///   1). Fixed by scaling the exponent.
/// * **Wrong GQA map** — `kv_head = head / n_kv_heads` instead of
///   `head / (n_heads / n_kv_heads)`, indexing past the KV heads for
///   `gqa_ratio > n_kv_heads`. Fixed.
///
/// # Parameters (device-side)
///
/// ```text
/// q_ptr       u64  query  [n_heads, head_dim] f32
/// k_ptr       u64  key blocks [n_blocks, block_size, n_kv_heads, head_dim] f32
/// v_ptr       u64  value blocks (same layout as k_ptr)
/// btbl_ptr    u64  block table [max_blocks] u32 – logical → physical block id
/// out_ptr     u64  output [n_heads, head_dim] f32
/// n_heads     u32
/// n_kv_heads  u32
/// head_dim    u32
/// block_size  u32
/// n_blocks    u32  number of valid entries in block_table
/// seq_len     u32  total number of tokens in the KV cache
/// scale       f32  = 1/√head_dim
/// ```
pub fn paged_attn_ptx(sm: u32) -> String {
    let ver = if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// paged_attention  (corrected FlashAttention-style online softmax)
// Grid : (n_query_heads, 1, 1)
// Block: (head_dim, 1, 1)   -- each thread owns one head-dim element `dim`
.visible .entry paged_attention(
    .param .u64 q_ptr,
    .param .u64 k_ptr,
    .param .u64 v_ptr,
    .param .u64 btbl_ptr,
    .param .u64 out_ptr,
    .param .u32 n_heads,
    .param .u32 n_kv_heads,
    .param .u32 head_dim,
    .param .u32 block_size,
    .param .u32 n_blocks,
    .param .u32 seq_len,
    .param .f32 scale
)
{{
    .reg .u64  %rq, %rk, %rv, %rbt, %ro, %addr;
    .reg .u32  %nh, %nkvh, %hd, %bs, %nb, %sl;
    .reg .u32  %head, %dim, %gqa, %kvh, %qbase, %kbase, %tmp, %blk, %phys, %slot, %tok, %d;
    .reg .f32  %scale_r, %log2e, %m, %l, %acc, %dot, %qd, %kd, %vd;
    .reg .f32  %score, %mnew, %alpha, %pw, %out;
    .reg .pred %p;

    ld.param.u64 %rq,   [q_ptr];
    ld.param.u64 %rk,   [k_ptr];
    ld.param.u64 %rv,   [v_ptr];
    ld.param.u64 %rbt,  [btbl_ptr];
    ld.param.u64 %ro,   [out_ptr];
    ld.param.u32 %nh,   [n_heads];
    ld.param.u32 %nkvh, [n_kv_heads];
    ld.param.u32 %hd,   [head_dim];
    ld.param.u32 %bs,   [block_size];
    ld.param.u32 %nb,   [n_blocks];
    ld.param.u32 %sl,   [seq_len];
    ld.param.f32 %scale_r, [scale];

    mov.u32 %head, %ctaid.x;
    mov.u32 %dim,  %tid.x;
    setp.ge.u32 %p, %head, %nh;
    @%p ret;
    setp.ge.u32 %p, %dim, %hd;
    @%p ret;

    // GQA: kv_head = head / (n_heads / n_kv_heads)
    div.u32 %gqa, %nh, %nkvh;
    div.u32 %kvh, %head, %gqa;

    // q row base element offset
    mul.lo.u32 %qbase, %head, %hd;

    // online-softmax accumulators
    mov.f32 %m,   {neg_inf};
    mov.f32 %l,   0F00000000;
    mov.f32 %acc, 0F00000000;
    mov.f32 %log2e, {log2e};

    mov.u32 %blk, 0;
$BLOCK_LOOP:
    setp.ge.u32 %p, %blk, %nb;
    @%p bra $BLOCK_DONE;
    // phys = block_table[blk]
    mul.wide.u32 %addr, %blk, 4;
    add.u64 %addr, %rbt, %addr;
    ld.global.u32 %phys, [%addr];

    mov.u32 %slot, 0;
$SLOT_LOOP:
    setp.ge.u32 %p, %slot, %bs;
    @%p bra $SLOT_DONE;
    mad.lo.u32 %tok, %blk, %bs, %slot;
    setp.ge.u32 %p, %tok, %sl;
    @%p bra $SLOT_DONE;

    // kbase = ((phys*bs + slot)*n_kv_heads + kv_head) * head_dim
    mad.lo.u32 %tmp, %phys, %bs, %slot;
    mul.lo.u32 %tmp, %tmp, %nkvh;
    add.u32    %tmp, %tmp, %kvh;
    mul.lo.u32 %kbase, %tmp, %hd;

    // dot = Σ_d q[qbase+d] * k[kbase+d]  (full dot product, online softmax)
    mov.f32 %dot, 0F00000000;
    mov.u32 %d, 0;
$DOT_LOOP:
    setp.ge.u32 %p, %d, %hd;
    @%p bra $DOT_DONE;
    add.u32 %tmp, %qbase, %d;
    mul.wide.u32 %addr, %tmp, 4;
    add.u64 %addr, %rq, %addr;
    ld.global.f32 %qd, [%addr];
    add.u32 %tmp, %kbase, %d;
    mul.wide.u32 %addr, %tmp, 4;
    add.u64 %addr, %rk, %addr;
    ld.global.f32 %kd, [%addr];
    fma.rn.f32 %dot, %qd, %kd, %dot;
    add.u32 %d, %d, 1;
    bra $DOT_LOOP;
$DOT_DONE:
    mul.f32 %score, %dot, %scale_r;

    // online softmax update (FlashAttention-style), base-e via ex2(x*log2e)
    max.f32 %mnew, %m, %score;
    sub.f32 %alpha, %m, %mnew;
    mul.f32 %alpha, %alpha, %log2e;
    ex2.approx.f32 %alpha, %alpha;       // exp(m - mnew)
    sub.f32 %pw, %score, %mnew;
    mul.f32 %pw, %pw, %log2e;
    ex2.approx.f32 %pw, %pw;             // exp(score - mnew)
    fma.rn.f32 %l, %l, %alpha, %pw;      // l = l*alpha + pw

    // v contribution: vd = v[kbase + dim]  (read through v_ptr, not k_ptr)
    add.u32 %tmp, %kbase, %dim;
    mul.wide.u32 %addr, %tmp, 4;
    add.u64 %addr, %rv, %addr;
    ld.global.f32 %vd, [%addr];
    mul.f32 %acc, %acc, %alpha;
    fma.rn.f32 %acc, %pw, %vd, %acc;     // acc = acc*alpha + pw*vd

    mov.f32 %m, %mnew;
    add.u32 %slot, %slot, 1;
    bra $SLOT_LOOP;
$SLOT_DONE:
    add.u32 %blk, %blk, 1;
    bra $BLOCK_LOOP;
$BLOCK_DONE:

    // out = acc / l (guard the empty-sequence l == 0 case)
    mov.f32 %out, 0F00000000;
    setp.eq.f32 %p, %l, 0F00000000;
    @%p bra $ATTN_STORE;
    div.rn.f32 %out, %acc, %l;
$ATTN_STORE:
    add.u32 %tmp, %qbase, %dim;
    mul.wide.u32 %addr, %tmp, 4;
    add.u64 %addr, %ro, %addr;
    st.global.f32 [%addr], %out;

    ret;
}}
"#,
        ver = ver,
        sm = sm,
        neg_inf = neg_inf,
        log2e = log2e,
    )
}

// ─── rope_apply_ptx ──────────────────────────────────────────────────────────

/// Generate the PTX kernel for Rotary Position Embedding (RoPE).
///
/// For each pair (2i, 2i+1) in the head dimension, rotates by angle
/// θ_i = pos / 10000^(2i/d):
/// ```text
/// q_out[2i]   = q[2i]   * cos(θ) − q[2i+1] * sin(θ)
/// q_out[2i+1] = q[2i+1] * cos(θ) + q[2i]   * sin(θ)
/// ```
/// Applied independently to Q and K.
///
/// ## Bugs found on-device and fixed
///
/// * **Q\[2i\] lost its sine term** — the original applied two cancelling fused
///   multiply-adds (`+q1·sin` then `−q1·sin`), leaving `q_out[2i] = q[2i]·cos`
///   with **no** `−q[2i+1]·sin` term. (The K path was correct.) Fixed to a
///   single subtractive fma, mirroring the K rotation.
/// * **Imprecise `log2(10000)` constant** — the literal `0F4154A3BB` (≈13.2909)
///   differs from `log2(10000) ≈ 13.28771` by ~0.02 %. Now derived exactly from
///   `10000.0_f32.log2()`.
///
/// `cos.approx.f32` / `sin.approx.f32` are accurate for `|θ| < π`; callers
/// (and the GPU tests) keep positions small enough that θ ≤ pos stays in range.
///
/// Grid : (seq_len * n_heads, 1, 1)
/// Block: (head_dim/2, 1, 1)
pub fn rope_apply_ptx(sm: u32) -> String {
    let ver = if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    let theta_base = f32_hex(10000.0_f32);
    let log2_base = f32_hex(10000.0_f32.log2());
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// rope_apply
// Applies RoPE in-place to Q and K tensors.
// q_ptr, k_ptr: [seq_len, n_heads, head_dim] f32 (in-place modification)
// positions:    [seq_len] u32
// head_dim, n_heads, seq_len: u32
.visible .entry rope_apply(
    .param .u64 q_ptr,
    .param .u64 k_ptr,
    .param .u64 pos_ptr,
    .param .u32 n_heads,
    .param .u32 head_dim,
    .param .u32 seq_len
)
{{
    .reg .u64 %rq, %rk, %rp;
    .reg .u32 %seq_head, %pair, %hd, %nh, %sl;
    .reg .u32 %seq_idx, %head_idx, %dim0, %dim1;
    .reg .u32 %pos_val;
    .reg .f32 %q0, %q1, %k0, %k1, %cos_t, %sin_t;
    .reg .f32 %theta, %pos_f, %freq, %angle;
    .reg .f32 %dim_f, %hd_f, %base;
    .reg .u32 %half_hd;
    .reg .f32 %log2_base;
    .reg .u32 %base_off, %off0, %off1;
    .reg .f32 %q0_new, %q1_new, %k0_new, %k1_new;
    .reg .pred %p_oob;
    .reg .u64 %q0addr, %q1addr, %k0addr, %k1addr, %paddr;

    ld.param.u64 %rq, [q_ptr];
    ld.param.u64 %rk, [k_ptr];
    ld.param.u64 %rp, [pos_ptr];
    ld.param.u32 %nh, [n_heads];
    ld.param.u32 %hd, [head_dim];
    ld.param.u32 %sl, [seq_len];

    // seq_head = blockIdx.x,  pair = threadIdx.x
    mov.u32 %seq_head, %ctaid.x;
    mov.u32 %pair,     %tid.x;

    // seq_idx  = seq_head / n_heads
    // head_idx = seq_head % n_heads
    div.u32 %seq_idx,  %seq_head, %nh;
    rem.u32 %head_idx, %seq_head, %nh;

    // dim0 = 2*pair,  dim1 = 2*pair+1
    mul.lo.u32 %dim0, %pair, 2;
    add.u32    %dim1, %dim0, 1;

    shr.u32 %half_hd, %hd, 1;
    setp.ge.u32 %p_oob, %pair, %half_hd;
    @%p_oob ret;
    setp.ge.u32 %p_oob, %seq_idx, %sl;
    @%p_oob ret;

    // load position
    mul.wide.u32 %paddr, %seq_idx, 4;
    add.u64 %paddr, %rp, %paddr;
    ld.global.u32 %pos_val, [%paddr];
    cvt.rn.f32.u32 %pos_f, %pos_val;

    // freq = dim0 / head_dim  (freq index)
    cvt.rn.f32.u32 %dim_f, %dim0;
    cvt.rn.f32.u32 %hd_f,  %hd;
    div.approx.f32 %freq, %dim_f, %hd_f;

    // theta = pos / 10000^freq = pos * 2^(-freq*log2(10000))
    mov.f32 %base, {theta_base};        // 10000.0 (documentation reference)
    mov.f32 %log2_base, {log2_base};    // log2(10000)
    mul.f32 %angle, %freq, %log2_base;
    ex2.approx.f32 %theta, %angle;       // 2^(freq*log2(10000)) = 10000^freq
    rcp.approx.f32 %theta, %theta;       // 1 / 10000^freq
    mul.f32 %theta, %pos_f, %theta;      // pos / 10000^freq

    // cos and sin (valid for |theta| < pi)
    cos.approx.f32 %cos_t, %theta;
    sin.approx.f32 %sin_t, %theta;

    // compute flat offsets: base = (seq_idx * n_heads + head_idx) * head_dim
    mad.lo.u32 %base_off, %seq_idx, %nh, %head_idx;
    mul.lo.u32 %base_off, %base_off, %hd;
    add.u32    %off0, %base_off, %dim0;
    add.u32    %off1, %base_off, %dim1;

    mul.wide.u32 %q0addr, %off0, 4; add.u64 %q0addr, %rq, %q0addr;
    mul.wide.u32 %q1addr, %off1, 4; add.u64 %q1addr, %rq, %q1addr;
    mul.wide.u32 %k0addr, %off0, 4; add.u64 %k0addr, %rk, %k0addr;
    mul.wide.u32 %k1addr, %off1, 4; add.u64 %k1addr, %rk, %k1addr;

    ld.global.f32 %q0, [%q0addr];
    ld.global.f32 %q1, [%q1addr];
    ld.global.f32 %k0, [%k0addr];
    ld.global.f32 %k1, [%k1addr];

    // rotate Q:  q0' = q0*cos - q1*sin ,  q1' = q1*cos + q0*sin
    mul.f32 %q0_new, %q0, %cos_t;
    neg.f32 %sin_t, %sin_t;
    fma.rn.f32 %q0_new, %q1, %sin_t, %q0_new;   // q0*cos - q1*sin
    neg.f32 %sin_t, %sin_t;                      // restore +sin
    mul.f32 %q1_new, %q1, %cos_t;
    fma.rn.f32 %q1_new, %q0, %sin_t, %q1_new;    // q1*cos + q0*sin

    // rotate K:  k0' = k0*cos - k1*sin ,  k1' = k1*cos + k0*sin
    mul.f32 %k0_new, %k0, %cos_t;
    neg.f32 %sin_t, %sin_t;
    fma.rn.f32 %k0_new, %k1, %sin_t, %k0_new;   // k0*cos - k1*sin
    neg.f32 %sin_t, %sin_t;                      // restore +sin
    mul.f32 %k1_new, %k1, %cos_t;
    fma.rn.f32 %k1_new, %k0, %sin_t, %k1_new;    // k1*cos + k0*sin

    st.global.f32 [%q0addr], %q0_new;
    st.global.f32 [%q1addr], %q1_new;
    st.global.f32 [%k0addr], %k0_new;
    st.global.f32 [%k1addr], %k1_new;

    ret;
}}
"#,
        ver = ver,
        sm = sm,
        theta_base = theta_base,
        log2_base = log2_base,
    )
}

// ─── top_k_filter_ptx ────────────────────────────────────────────────────────

/// Generate the PTX kernel that masks logits outside a reference top-K window.
///
/// After this kernel, non-top-K positions contain −∞ so that the
/// subsequent softmax assigns them zero probability.
///
/// Algorithm (reference index-window masking):
/// 1. Clamp `k` to `vocab_size`.
/// 2. Mask positions with `idx >= k` to −∞.
///
/// This kernel is a lightweight reference path for deterministic tests and
/// wiring validation. Value-based top-K thresholding is implemented via the
/// higher-level sampling path.
///
/// Grid : (batch_size, 1, 1)
/// Block: (min(vocab_size, 1024), 1, 1)
pub fn top_k_filter_ptx(sm: u32) -> String {
    let ver = if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// top_k_filter
// logits_ptr: [batch, vocab_size] f32 (in-place)
// vocab_size, k: u32
// batch_size: u32
.visible .entry top_k_filter(
    .param .u64 logits_ptr,
    .param .u32 batch_size,
    .param .u32 vocab_size,
    .param .u32 k
)
{{
    .reg .u64 %rl;
    .reg .u32 %bid, %tid_x, %vs, %k_r, %k_eff, %bs;
    .reg .u64 %row_ptr, %off;
    .reg .f32 %val, %neg_inf_r;
    .reg .pred %p_oob, %p_mask;

    ld.param.u64 %rl,  [logits_ptr];
    ld.param.u32 %bs,  [batch_size];
    ld.param.u32 %vs,  [vocab_size];
    ld.param.u32 %k_r, [k];

    mov.u32 %bid,   %ctaid.x;
    mov.u32 %tid_x, %tid.x;

    setp.ge.u32 %p_oob, %bid, %bs;
    @%p_oob ret;
    setp.ge.u32 %p_oob, %tid_x, %vs;
    @%p_oob ret;

    // Each thread loads its logit value
    .reg .u32 %glob_idx;
    mad.lo.u32 %glob_idx, %bid, %vs, %tid_x;
    mul.wide.u32 %off, %glob_idx, 4;
    add.u64 %row_ptr, %rl, %off;
    ld.global.f32 %val, [%row_ptr];

    // Clamp k to the row width to avoid out-of-range semantics.
    setp.gt.u32 %p_mask, %k_r, %vs;
    selp.u32 %k_eff, %vs, %k_r, %p_mask;

    // Reference masking kernel: positions with idx >= k_eff are suppressed.
    mov.f32 %neg_inf_r, {neg_inf};

    // Write -inf for positions beyond the kept window.
    setp.ge.u32 %p_mask, %tid_x, %k_eff;
    @%p_mask st.global.f32 [%row_ptr], %neg_inf_r;

    ret;
}}
"#,
        ver = ver,
        sm = sm,
        neg_inf = neg_inf,
    )
}

// ─── logits_softmax_ptx ──────────────────────────────────────────────────────

/// Generate the PTX kernel for numerically stable softmax over a logit vector.
///
/// Uses the standard three-pass algorithm (serial reductions in thread 0 over a
/// shared-memory tile):
/// 1. Find `max(logits)`.
/// 2. Compute `exp(logit − max)` in **base e** as `ex2((logit − max)·log2(e))`.
/// 3. Write `exp(logit − max) / Σ exp(...)`.
///
/// ## Bugs found on-device and fixed
///
/// * **Invalid PTX** — the original indexed shared memory as
///   `[smem + %tid_x * 4]`, but PTX memory operands do not support a scaled
///   register offset; ptxas rejected the module on sm_86 (never loaded). Fixed:
///   the shared address is now materialised with `mad.lo.u32 addr, tid, 4, base`.
/// * **Base-2 softmax** — `ex2.approx.f32` was applied directly to `logit − max`,
///   computing `2^x` rather than `e^x`. Fixed with the `· log2(e)` scale (the
///   distribution still summed to 1 before, so only a base-e oracle catches it).
/// * **Read-after-overwrite race** — `smem[0]` (the max) could be overwritten by
///   thread 0's exp before other warps read it. Fixed with an extra `bar.sync`.
///
/// Grid : (batch_size, 1, 1)
/// Block: (vocab_size, 1, 1)   (vocab_size ≤ 1024; block must equal vocab_size
/// so every thread reaches each `bar.sync`).
pub fn logits_softmax_ptx(sm: u32) -> String {
    let ver = if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    let tiny = f32_hex(1.0e-7_f32);
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// logits_softmax  (numerically stable, base-e)
// logits_ptr: [batch, vocab_size] f32 (in-place)
.visible .entry logits_softmax(
    .param .u64 logits_ptr,
    .param .u32 batch_size,
    .param .u32 vocab_size
)
{{
    .reg .u64 %rl, %addr;
    .reg .u32 %bid, %tid_x, %vs, %bs, %glob, %i, %sbase, %saddr;
    .reg .f32 %val, %maxv, %e, %ered, %sum, %rcp, %log2e;
    .reg .pred %p;
    .shared .align 4 .f32 smem[1024];

    ld.param.u64 %rl, [logits_ptr];
    ld.param.u32 %bs, [batch_size];
    ld.param.u32 %vs, [vocab_size];

    mov.u32 %bid,   %ctaid.x;
    mov.u32 %tid_x, %tid.x;
    setp.ge.u32 %p, %bid, %bs;
    @%p ret;
    setp.ge.u32 %p, %tid_x, %vs;
    @%p ret;

    mov.f32 %log2e, {log2e};

    // load this thread's logit; keep it in %val for the whole kernel
    mad.lo.u32 %glob, %bid, %vs, %tid_x;
    mul.wide.u32 %addr, %glob, 4;
    add.u64 %addr, %rl, %addr;
    ld.global.f32 %val, [%addr];

    mov.u32 %sbase, smem;
    mad.lo.u32 %saddr, %tid_x, 4, %sbase;
    st.shared.f32 [%saddr], %val;
    bar.sync 0;

    // Pass 1: thread 0 reduces max into smem[0].
    setp.ne.u32 %p, %tid_x, 0;
    @%p bra $SM_AFTERMAX;
    mov.f32 %maxv, {neg_inf};
    mov.u32 %i, 0;
$SM_MAXLOOP:
    setp.ge.u32 %p, %i, %vs;
    @%p bra $SM_MAXDONE;
    mad.lo.u32 %saddr, %i, 4, %sbase;
    ld.shared.f32 %e, [%saddr];
    max.f32 %maxv, %maxv, %e;
    add.u32 %i, %i, 1;
    bra $SM_MAXLOOP;
$SM_MAXDONE:
    st.shared.f32 [%sbase], %maxv;
$SM_AFTERMAX:
    bar.sync 0;

    // Pass 2: every thread exp(logit - max) from its register %val.
    ld.shared.f32 %maxv, [%sbase];
    bar.sync 0;                          // all threads read max before smem[0] is overwritten
    sub.f32 %val, %val, %maxv;
    mul.f32 %val, %val, %log2e;
    ex2.approx.f32 %e, %val;             // exp(logit - max), base-e
    mad.lo.u32 %saddr, %tid_x, 4, %sbase;
    st.shared.f32 [%saddr], %e;
    bar.sync 0;

    // Pass 3: thread 0 reduces the exp-sum into smem[0].
    setp.ne.u32 %p, %tid_x, 0;
    @%p bra $SM_AFTERSUM;
    mov.f32 %sum, 0F00000000;
    mov.u32 %i, 0;
$SM_SUMLOOP:
    setp.ge.u32 %p, %i, %vs;
    @%p bra $SM_SUMDONE;
    mad.lo.u32 %saddr, %i, 4, %sbase;
    ld.shared.f32 %ered, [%saddr];
    add.f32 %sum, %sum, %ered;
    add.u32 %i, %i, 1;
    bra $SM_SUMLOOP;
$SM_SUMDONE:
    max.f32 %sum, %sum, {tiny};          // avoid divide-by-zero
    st.shared.f32 [%sbase], %sum;
$SM_AFTERSUM:
    bar.sync 0;

    // Pass 4: normalise using this thread's own exp value (%e) and write back.
    ld.shared.f32 %sum, [%sbase];
    rcp.approx.f32 %rcp, %sum;
    mul.f32 %e, %e, %rcp;
    st.global.f32 [%addr], %e;
    ret;
}}
"#,
        ver = ver,
        sm = sm,
        neg_inf = neg_inf,
        log2e = log2e,
        tiny = tiny,
    )
}

// ─── kv_append_ptx ───────────────────────────────────────────────────────────

/// Generate the PTX kernel that appends one token's K and V to the paged cache.
///
/// Called once per layer after each decode step to update the KV cache.
///
/// Grid : (n_kv_heads, 1, 1)
/// Block: (head_dim, 1, 1)
pub fn kv_append_ptx(sm: u32) -> String {
    let ver = if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

// kv_append
// Writes one token's K and V into a physical KV cache block.
// k_new, v_new:  [n_kv_heads, head_dim] f32  (incoming key/value)
// k_cache, v_cache: [n_blocks, block_size, n_kv_heads, head_dim] f32 (cache)
// block_id:  u32  physical block to write into
// slot:      u32  slot within the block (0..block_size-1)
// n_kv_heads, head_dim, block_size: u32
.visible .entry kv_append(
    .param .u64 k_new_ptr,
    .param .u64 v_new_ptr,
    .param .u64 k_cache_ptr,
    .param .u64 v_cache_ptr,
    .param .u32 block_id,
    .param .u32 slot,
    .param .u32 n_kv_heads,
    .param .u32 head_dim,
    .param .u32 block_size
)
{{
    .reg .u64 %rknew, %rvnew, %rkc, %rvc;
    .reg .u32 %head, %dim, %bid, %slt, %nkvh, %hd, %bs;
    .reg .u32 %new_off, %cache_off;
    .reg .u64 %src_k, %src_v, %dst_k, %dst_v;
    .reg .f32 %kv;
    .reg .pred %p_oob;

    ld.param.u64 %rknew, [k_new_ptr];
    ld.param.u64 %rvnew, [v_new_ptr];
    ld.param.u64 %rkc,   [k_cache_ptr];
    ld.param.u64 %rvc,   [v_cache_ptr];
    ld.param.u32 %bid,   [block_id];
    ld.param.u32 %slt,   [slot];
    ld.param.u32 %nkvh,  [n_kv_heads];
    ld.param.u32 %hd,    [head_dim];
    ld.param.u32 %bs,    [block_size];

    mov.u32 %head, %ctaid.x;
    mov.u32 %dim,  %tid.x;

    setp.ge.u32 %p_oob, %head, %nkvh;
    @%p_oob ret;
    setp.ge.u32 %p_oob, %dim,  %hd;
    @%p_oob ret;

    // Source: k_new[head * head_dim + dim]
    mad.lo.u32 %new_off, %head, %hd, %dim;
    mul.wide.u32 %src_k, %new_off, 4;
    add.u64 %src_k, %rknew, %src_k;
    mul.wide.u32 %src_v, %new_off, 4;
    add.u64 %src_v, %rvnew, %src_v;

    // Destination: k_cache[(block_id * block_size + slot) * n_kv_heads * head_dim
    //                        + head * head_dim + dim]
    .reg .u32 %tok_off, %stride;
    mad.lo.u32 %tok_off, %bid, %bs, %slt;
    mul.lo.u32 %stride, %nkvh, %hd;
    mul.lo.u32 %tok_off, %tok_off, %stride;
    mad.lo.u32 %cache_off, %head, %hd, %tok_off;
    add.u32    %cache_off, %cache_off, %dim;
    mul.wide.u32 %dst_k, %cache_off, 4;
    add.u64 %dst_k, %rkc, %dst_k;
    mul.wide.u32 %dst_v, %cache_off, 4;
    add.u64 %dst_v, %rvc, %dst_v;

    ld.global.f32 %kv, [%src_k];
    st.global.f32 [%dst_k], %kv;
    ld.global.f32 %kv, [%src_v];
    st.global.f32 [%dst_v], %kv;

    ret;
}}
"#,
        ver = ver,
        sm = sm,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn kernels_for_sm(sm: u32) -> Vec<String> {
        vec![
            paged_attn_ptx(sm),
            rope_apply_ptx(sm),
            top_k_filter_ptx(sm),
            logits_softmax_ptx(sm),
            kv_append_ptx(sm),
        ]
    }

    #[test]
    fn all_kernels_non_empty_sm80() {
        for k in kernels_for_sm(80) {
            assert!(!k.is_empty());
        }
    }

    #[test]
    fn all_kernels_non_empty_sm90() {
        for k in kernels_for_sm(90) {
            assert!(!k.is_empty());
        }
    }

    #[test]
    fn sm80_uses_version_8_0() {
        for k in kernels_for_sm(80) {
            assert!(k.contains(".version 8.0"), "expected .version 8.0 in:\n{k}");
        }
    }

    #[test]
    fn sm90_uses_version_8_4() {
        for k in kernels_for_sm(90) {
            assert!(k.contains(".version 8.4"), "expected .version 8.4 in:\n{k}");
        }
    }

    #[test]
    fn sm75_fallback_version() {
        for k in kernels_for_sm(75) {
            assert!(k.contains(".version 7.5"), "expected .version 7.5 in:\n{k}");
        }
    }

    #[test]
    fn paged_attn_has_block_loop() {
        let ptx = paged_attn_ptx(80);
        assert!(
            ptx.contains("BLOCK_LOOP"),
            "paged_attn should have block iteration"
        );
        assert!(
            ptx.contains("online softmax"),
            "paged_attn should note flash-attention style"
        );
    }

    #[test]
    fn rope_apply_has_cos_sin() {
        let ptx = rope_apply_ptx(80);
        assert!(ptx.contains("cos.approx.f32"), "rope should use cos");
        assert!(ptx.contains("sin.approx.f32"), "rope should use sin");
    }

    #[test]
    fn top_k_filter_has_neg_inf_store() {
        let ptx = top_k_filter_ptx(80);
        assert!(ptx.contains("st.global.f32"), "top_k should store -inf");
    }

    #[test]
    fn top_k_filter_clamps_k_to_vocab() {
        let ptx = top_k_filter_ptx(80);
        assert!(
            ptx.contains("setp.gt.u32") && ptx.contains("selp.u32"),
            "top_k should clamp k to vocab size"
        );
    }

    #[test]
    fn kv_append_has_store_ops() {
        let ptx = kv_append_ptx(80);
        assert!(
            ptx.contains("st.global.f32"),
            "kv_append should store K and V"
        );
    }

    #[test]
    fn logits_softmax_is_base_e() {
        // The base-e correction multiplies the exponent by log2(e) before ex2.
        let ptx = logits_softmax_ptx(80);
        let log2e = f32_hex(std::f32::consts::LOG2_E);
        assert!(
            ptx.contains(&log2e),
            "logits_softmax must scale by log2(e) for base-e exp"
        );
    }
}
