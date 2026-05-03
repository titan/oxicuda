//! PTX GPU kernel sources for vision model operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`patch_embed_ptx`] | Strided Conv2D for ViT patch embedding CHW → patches×embed |
//! | [`bilinear_interp_ptx`] | Sub-pixel bilinear image resampling |
//! | [`contrastive_loss_ptx`] | InfoNCE/NT-Xent per-row cross-entropy over similarity matrix |
//! | [`roi_align_ptx`] | Per-bin bilinear RoI feature extraction (Faster R-CNN) |
//! | [`image_normalize_ptx`] | Channel-wise `(x - mean[c]) / std[c]` in-place over CHW |
//! | [`adaptive_avg_pool_ptx`] | Adaptive 2D average pooling: variable→fixed output size |
//! | [`focal_loss_ptx`] | Focal loss: −α·(1−p)^γ·log(p) for object detection |

// ─── Hex encoding ────────────────────────────────────────────────────────────

/// Encode a `f32` as a PTX hexadecimal float literal (e.g., `0F3F800000` = 1.0f).
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── PTX header helper ───────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    let ptx_ver = if sm >= 100 {
        "8.7"
    } else if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    format!(".version {ptx_ver}\n.target sm_{sm}\n.address_size 64\n\n")
}

// ─── Kernel 1: patch_embed ───────────────────────────────────────────────────

/// Strided 2D Conv2D for ViT patch embedding.
///
/// Converts a `[in_chans, img_size, img_size]` input image to
/// `[n_patches, embed_dim]` patch tokens by applying a
/// `[embed_dim, in_chans, patch_size, patch_size]` kernel with stride = patch_size.
///
/// One thread handles one `(patch_idx, embed_dim)` output element.
/// The flat tid encodes `patch_idx * embed_dim + e`.
///
/// Inner loop iterates over `in_chans × patch_size × patch_size` elements,
/// accumulating with `fma.rn.f32`.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_in` | `u64` (→ `f32*`) | Input `[in_chans × img_size × img_size]` |
/// | `p_kernel` | `u64` (→ `f32*`) | Conv kernel `[embed_dim × in_chans × patch_size × patch_size]` |
/// | `p_bias` | `u64` (→ `f32*`) | Bias `[embed_dim]` |
/// | `p_out` | `u64` (→ `f32*`) | Output `[n_patches × embed_dim]` |
/// | `n_patches` | `u32` | Total number of patches = `(img_size/patch_size)²` |
/// | `embed_dim` | `u32` | Embedding / output channel dimension |
/// | `in_chans` | `u32` | Number of input channels |
/// | `patch_size` | `u32` | Spatial patch size P (kernel = P×P) |
/// | `img_size` | `u32` | Square image spatial dimension (H = W) |
///
/// Launch: `grid = ceil(n_patches * embed_dim / 256)`, `block = 256`.
#[must_use]
pub fn patch_embed_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry patch_embed(
    .param .u64 p_in,
    .param .u64 p_kernel,
    .param .u64 p_bias,
    .param .u64 p_out,
    .param .u32 n_patches,
    .param .u32 embed_dim,
    .param .u32 in_chans,
    .param .u32 patch_size,
    .param .u32 img_size
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<32>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_in];
    ld.param.u64  %rd1, [p_kernel];
    ld.param.u64  %rd2, [p_bias];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [n_patches];
    ld.param.u32  %r1,  [embed_dim];
    ld.param.u32  %r2,  [in_chans];
    ld.param.u32  %r3,  [patch_size];
    ld.param.u32  %r4,  [img_size];

    // Grid-stride: one thread per (patch_idx, embed) pair
    // flat_tid = patch_idx * embed_dim + e
    mov.u32       %r5, %ntid.x;
    mov.u32       %r6, %ctaid.x;
    mov.u32       %r7, %tid.x;
    mad.lo.u32    %r8, %r5, %r6, %r7;     // r8 = flat_tid

    // total = n_patches * embed_dim
    mul.lo.u32    %r9, %r0, %r1;

$PE_OUTER:
    setp.ge.u32   %p0, %r8, %r9;
    @%p0 bra $PE_DONE;

    // patch_idx = flat_tid / embed_dim
    // e         = flat_tid % embed_dim
    div.u32       %r10, %r8, %r1;         // r10 = patch_idx
    rem.u32       %r11, %r8, %r1;         // r11 = e (embed channel)

    // grid_w = img_size / patch_size  (number of patches per row)
    div.u32       %r12, %r4, %r3;         // r12 = grid_w

    // patch row and column
    div.u32       %r13, %r10, %r12;       // r13 = ph = patch_idx / grid_w
    rem.u32       %r14, %r10, %r12;       // r14 = pw = patch_idx % grid_w

    // top-left pixel of this patch in the full image
    // ph_start = ph * patch_size,  pw_start = pw * patch_size
    mul.lo.u32    %r15, %r13, %r3;        // r15 = ph_start
    mul.lo.u32    %r16, %r14, %r3;        // r16 = pw_start

    // Load bias[e]
    mul.wide.u32  %rd4, %r11, 4;
    add.u64       %rd5, %rd2, %rd4;
    ld.global.f32 %f0, [%rd5];            // f0 = acc = bias[e]

    // kernel_stride_e = in_chans * patch_size * patch_size
    // kernel base offset for embed e: e * in_chans * patch_size * patch_size * 4
    mul.lo.u32    %r17, %r2, %r3;         // in_chans * patch_size
    mul.lo.u32    %r17, %r17, %r3;        // in_chans * patch_size * patch_size
    mul.lo.u32    %r18, %r11, %r17;       // r18 = kernel_e_base (elem offset)
    mul.wide.u32  %rd6, %r18, 4;
    add.u64       %rd7, %rd1, %rd6;       // rd7 = kernel ptr for embed e

    // Inner loop: c in [0, in_chans), ky in [0, patch_size), kx in [0, patch_size)
    // loop variable: r19 = c,  r20 = ky,  r21 = kx
    mov.u32       %r19, 0;                // c = 0

$PE_CLOOP:
    setp.ge.u32   %p1, %r19, %r2;
    @%p1 bra $PE_CEND;

    mov.u32       %r20, 0;                // ky = 0

$PE_KYLOOP:
    setp.ge.u32   %p1, %r20, %r3;
    @%p1 bra $PE_KYEND;

    mov.u32       %r21, 0;               // kx = 0

$PE_KXLOOP:
    setp.ge.u32   %p1, %r21, %r3;
    @%p1 bra $PE_KXEND;

    // Input pixel: img[c, ph_start + ky, pw_start + kx]
    // img offset = (c * img_size + ph_start + ky) * img_size + pw_start + kx
    add.u32       %r22, %r15, %r20;      // ph_start + ky
    mad.lo.u32    %r22, %r19, %r4, %r22; // c * img_size + (ph_start + ky)
    mul.lo.u32    %r22, %r22, %r4;       // * img_size
    add.u32       %r22, %r22, %r16;      // + pw_start
    add.u32       %r22, %r22, %r21;      // + kx
    mul.wide.u32  %rd8, %r22, 4;
    add.u64       %rd9, %rd0, %rd8;
    ld.global.f32 %f1, [%rd9];           // f1 = input pixel

    // Kernel weight: kernel[e, c, ky, kx]
    // offset from rd7 (already at embed e base):
    // (c * patch_size + ky) * patch_size + kx
    mad.lo.u32    %r23, %r19, %r3, %r20; // c * patch_size + ky
    mul.lo.u32    %r23, %r23, %r3;       // * patch_size
    add.u32       %r23, %r23, %r21;      // + kx
    mul.wide.u32  %rd10, %r23, 4;
    add.u64       %rd11, %rd7, %rd10;
    ld.global.f32 %f2, [%rd11];          // f2 = kernel weight

    fma.rn.f32    %f0, %f2, %f1, %f0;   // acc += w * x

    add.u32       %r21, %r21, 1;
    bra           $PE_KXLOOP;

$PE_KXEND:
    add.u32       %r20, %r20, 1;
    bra           $PE_KYLOOP;

$PE_KYEND:
    add.u32       %r19, %r19, 1;
    bra           $PE_CLOOP;

$PE_CEND:
    // Store out[patch_idx, e]
    mul.wide.u32  %rd12, %r8, 4;
    add.u64       %rd13, %rd3, %rd12;
    st.global.f32 [%rd13], %f0;

    // Grid stride: advance by blockDim * gridDim
    mov.u32       %r5, %ntid.x;
    mov.u32       %r24, %nctaid.x;
    mul.lo.u32    %r24, %r5, %r24;
    add.u32       %r8, %r8, %r24;
    bra           $PE_OUTER;

$PE_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 2: bilinear_interp ───────────────────────────────────────────────

/// Sub-pixel bilinear image resampler.
///
/// Resamples a `[n_channels, in_h, in_w]` source to `[n_channels, out_h, out_w]`
/// using the half-pixel convention:
/// `src_y = (oy + 0.5) × (in_h / out_h) − 0.5`, clamped to `[0, in_h − 1]`.
/// Similarly for x.  Four-tap bilinear interpolation is then performed.
///
/// One thread handles one output `(channel, oy, ox)` triple.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_src` | `u64` (→ `f32*`) | Source image `[n_chans × in_h × in_w]` |
/// | `p_out` | `u64` (→ `f32*`) | Output image `[n_chans × out_h × out_w]` |
/// | `in_h` | `u32` | Input height |
/// | `in_w` | `u32` | Input width |
/// | `out_h` | `u32` | Output height |
/// | `out_w` | `u32` | Output width |
/// | `n_chans` | `u32` | Number of channels |
///
/// Launch: `grid = ceil(n_chans * out_h * out_w / 256)`, `block = 256`.
#[must_use]
pub fn bilinear_interp_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let half = f32_hex(0.5_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}.visible .entry bilinear_interp(
    .param .u64 p_src,
    .param .u64 p_out,
    .param .u32 in_h,
    .param .u32 in_w,
    .param .u32 out_h,
    .param .u32 out_w,
    .param .u32 n_chans
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<24>;
    .reg .f32  %f<32>;
    .reg .pred %p0, %p1, %p2;

    ld.param.u64  %rd0, [p_src];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [in_h];
    ld.param.u32  %r1,  [in_w];
    ld.param.u32  %r2,  [out_h];
    ld.param.u32  %r3,  [out_w];
    ld.param.u32  %r4,  [n_chans];

    // Grid-stride: one thread per (c, oy, ox)
    // flat_tid = c * out_h * out_w + oy * out_w + ox
    mov.u32       %r5, %ntid.x;
    mov.u32       %r6, %ctaid.x;
    mov.u32       %r7, %tid.x;
    mad.lo.u32    %r8, %r5, %r6, %r7;    // r8 = flat_tid

    // total = n_chans * out_h * out_w
    mul.lo.u32    %r9, %r4, %r2;
    mul.lo.u32    %r9, %r9, %r3;

$BI_OUTER:
    setp.ge.u32   %p0, %r8, %r9;
    @%p0 bra $BI_DONE;

    // Decode: out_hw = out_h * out_w
    mul.lo.u32    %r10, %r2, %r3;         // r10 = out_hw
    div.u32       %r11, %r8, %r10;        // r11 = c
    rem.u32       %r12, %r8, %r10;        // r12 = oy * out_w + ox
    div.u32       %r13, %r12, %r3;        // r13 = oy
    rem.u32       %r14, %r12, %r3;        // r14 = ox

    // Compute src_y = (oy + 0.5) * (in_h / out_h) - 0.5
    // Use float arithmetic with cvt.rn
    cvt.rn.f32.u32  %f0, %r13;           // f0 = (f32)oy
    fma.rn.f32      %f0, %f0, {ONE}, {HALF}; // f0 = oy + 0.5
    cvt.rn.f32.u32  %f1, %r0;            // f1 = (f32)in_h
    cvt.rn.f32.u32  %f2, %r2;            // f2 = (f32)out_h
    div.rn.f32      %f3, %f1, %f2;       // f3 = in_h / out_h  (scale_y)
    mul.f32         %f4, %f0, %f3;       // f4 = (oy+0.5)*scale_y
    sub.f32         %f4, %f4, {HALF};    // f4 = src_y (before clamp)

    // Clamp src_y to [0, in_h - 1]
    max.f32         %f4, %f4, {ZERO};
    cvt.rn.f32.u32  %f5, %r0;            // f5 = (f32)in_h
    sub.f32         %f5, %f5, {ONE};     // f5 = in_h - 1.0
    min.f32         %f4, %f4, %f5;       // f4 = clamped src_y

    // floor(src_y) and frac_y
    floor.f32       %f6, %f4;            // f6 = y0 (float)
    sub.f32         %f7, %f4, %f6;       // f7 = fy = frac_y

    // Compute src_x = (ox + 0.5) * (in_w / out_w) - 0.5
    cvt.rn.f32.u32  %f8, %r14;           // f8 = (f32)ox
    fma.rn.f32      %f8, %f8, {ONE}, {HALF}; // f8 = ox + 0.5
    cvt.rn.f32.u32  %f9, %r1;            // f9 = (f32)in_w
    cvt.rn.f32.u32  %f10, %r3;           // f10 = (f32)out_w
    div.rn.f32      %f11, %f9, %f10;     // f11 = scale_x
    mul.f32         %f12, %f8, %f11;
    sub.f32         %f12, %f12, {HALF};  // f12 = src_x (before clamp)

    // Clamp src_x to [0, in_w - 1]
    max.f32         %f12, %f12, {ZERO};
    cvt.rn.f32.u32  %f13, %r1;
    sub.f32         %f13, %f13, {ONE};
    min.f32         %f12, %f12, %f13;

    // floor(src_x) and frac_x
    floor.f32       %f14, %f12;          // f14 = x0 (float)
    sub.f32         %f15, %f12, %f14;    // f15 = fx = frac_x

    // Convert floor coords to integers: y0, x0, y1 = min(y0+1, in_h-1), x1 = min(x0+1, in_w-1)
    cvt.rzi.u32.f32 %r15, %f6;           // r15 = y0
    cvt.rzi.u32.f32 %r16, %f14;          // r16 = x0

    // y1 = min(y0 + 1, in_h - 1)
    add.u32         %r17, %r15, 1;
    sub.u32         %r18, %r0, 1;        // in_h - 1
    min.u32         %r17, %r17, %r18;    // r17 = y1

    // x1 = min(x0 + 1, in_w - 1)
    add.u32         %r19, %r16, 1;
    sub.u32         %r20, %r1, 1;        // in_w - 1
    min.u32         %r19, %r19, %r20;    // r19 = x1

    // Channel base offset: c * in_h * in_w
    mul.lo.u32      %r21, %r11, %r0;     // c * in_h
    mul.lo.u32      %r21, %r21, %r1;     // c * in_h * in_w

    // Load 4 pixels: tl, tr, bl, br
    // tl = src[c, y0, x0]
    mad.lo.u32      %r22, %r15, %r1, %r16; // y0*in_w + x0
    add.u32         %r22, %r22, %r21;
    mul.wide.u32    %rd2, %r22, 4;
    add.u64         %rd3, %rd0, %rd2;
    ld.global.f32   %f16, [%rd3];        // f16 = tl

    // tr = src[c, y0, x1]
    mad.lo.u32      %r22, %r15, %r1, %r19; // y0*in_w + x1
    add.u32         %r22, %r22, %r21;
    mul.wide.u32    %rd2, %r22, 4;
    add.u64         %rd3, %rd0, %rd2;
    ld.global.f32   %f17, [%rd3];        // f17 = tr

    // bl = src[c, y1, x0]
    mad.lo.u32      %r22, %r17, %r1, %r16; // y1*in_w + x0
    add.u32         %r22, %r22, %r21;
    mul.wide.u32    %rd2, %r22, 4;
    add.u64         %rd3, %rd0, %rd2;
    ld.global.f32   %f18, [%rd3];        // f18 = bl

    // br = src[c, y1, x1]
    mad.lo.u32      %r22, %r17, %r1, %r19; // y1*in_w + x1
    add.u32         %r22, %r22, %r21;
    mul.wide.u32    %rd2, %r22, 4;
    add.u64         %rd3, %rd0, %rd2;
    ld.global.f32   %f19, [%rd3];        // f19 = br

    // Bilinear blend:
    // top = tl * (1 - fx) + tr * fx
    // bot = bl * (1 - fx) + br * fx
    // out = top * (1 - fy) + bot * fy
    sub.f32         %f20, {ONE}, %f15;   // 1 - fx
    sub.f32         %f21, {ONE}, %f7;    // 1 - fy

    mul.f32         %f22, %f16, %f20;    // tl * (1-fx)
    fma.rn.f32      %f22, %f17, %f15, %f22; // + tr * fx  (= top)

    mul.f32         %f23, %f18, %f20;    // bl * (1-fx)
    fma.rn.f32      %f23, %f19, %f15, %f23; // + br * fx  (= bot)

    mul.f32         %f24, %f22, %f21;    // top * (1-fy)
    fma.rn.f32      %f24, %f23, %f7, %f24; // + bot * fy  (= out pixel)

    // Store output
    mul.wide.u32    %rd4, %r8, 4;
    add.u64         %rd5, %rd1, %rd4;
    st.global.f32   [%rd5], %f24;

    // Grid stride
    mov.u32         %r5, %ntid.x;
    mov.u32         %r23, %nctaid.x;
    mul.lo.u32      %r23, %r5, %r23;
    add.u32         %r8, %r8, %r23;
    bra             $BI_OUTER;

$BI_DONE:
    ret;
}}
"#,
        ZERO = zero,
        HALF = half,
        ONE = one,
    )
}

// ─── Kernel 3: contrastive_loss ──────────────────────────────────────────────

/// InfoNCE / NT-Xent contrastive loss per row of a `[B, B]` similarity matrix.
///
/// Assumes similarity values have already been divided by temperature.
/// Implements a 3-pass numerically-stable softmax pattern per row:
///
/// ```text
/// pass1: max_val = max over j in [0,B) of sim[row, j]
/// pass2: sum_exp = Σ_j exp(sim[row,j] - max_val)
/// pass3: loss[row] = -(sim[row, row] - max_val) + log(sum_exp)
/// ```
///
/// For `B ≤ 32` each thread iterates sequentially over the `B` columns
/// of its row.  For large `B`, launch multiple thread-blocks and accumulate
/// externally; this kernel handles one row per thread.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_sim` | `u64` (→ `f32*`) | Similarity matrix `[n_batch × n_batch]` |
/// | `p_loss` | `u64` (→ `f32*`) | Per-row loss output `[n_batch]` |
/// | `n_batch` | `u32` | Batch size B |
///
/// Launch: `grid = ceil(n_batch / 256)`, `block = 256`.
#[must_use]
pub fn contrastive_loss_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    let zero = f32_hex(0.0_f32);
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    // ln(2) used to convert lg2 back to natural log
    let ln2 = f32_hex(std::f32::consts::LN_2);
    format!(
        r#"{hdr}.visible .entry contrastive_loss(
    .param .u64 p_sim,
    .param .u64 p_loss,
    .param .u32 n_batch
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<16>;
    .reg .f32  %f<16>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_sim];
    ld.param.u64  %rd1, [p_loss];
    ld.param.u32  %r0,  [n_batch];

    // Grid-stride: one thread per row
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;    // r4 = row index

$CL_OUTER:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $CL_DONE;

    // ── Pass 1: find row maximum ─────────────────────────────────────────────
    // base byte offset for row r4: r4 * n_batch * 4
    mul.lo.u32    %r5, %r4, %r0;          // r5 = row_base (elem)
    mov.u32       %r6, 0;                  // r6 = j (column)
    mov.f32       %f0, {NEG_INF};          // f0 = max_val

$CL_MAX_LOOP:
    setp.ge.u32   %p1, %r6, %r0;
    @%p1 bra $CL_MAX_END;

    add.u32       %r7, %r5, %r6;          // r5 + j
    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];            // f1 = sim[row, j]
    max.f32       %f0, %f0, %f1;          // update max

    add.u32       %r6, %r6, 1;
    bra           $CL_MAX_LOOP;

$CL_MAX_END:
    // ── Pass 2: sum of exp(sim - max_val) ───────────────────────────────────
    mov.u32       %r6, 0;
    mov.f32       %f2, {ZERO};            // f2 = sum_exp

$CL_SUM_LOOP:
    setp.ge.u32   %p1, %r6, %r0;
    @%p1 bra $CL_SUM_END;

    add.u32       %r7, %r5, %r6;
    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f3, [%rd3];            // sim[row, j]
    sub.f32       %f4, %f3, %f0;          // sim[row,j] - max_val
    mul.f32       %f5, %f4, {LOG2E};      // * log2(e)
    ex2.approx.f32 %f6, %f5;             // exp2(x) = exp(x * log2e)
    add.f32       %f2, %f2, %f6;          // sum_exp += exp(sim-max)

    add.u32       %r6, %r6, 1;
    bra           $CL_SUM_LOOP;

$CL_SUM_END:
    // ── Pass 3: loss[row] = -(sim[row,row] - max_val) + ln(sum_exp) ─────────
    // diagonal element: sim[row, row]
    add.u32       %r8, %r5, %r4;          // r5 (row_base) + r4 (row) = [row,row]
    mul.wide.u32  %rd2, %r8, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f7, [%rd3];            // f7 = sim[row, row]

    sub.f32       %f8, %f7, %f0;          // sim_diag - max_val
    neg.f32       %f9, %f8;               // -(sim_diag - max_val)

    // ln(sum_exp) = log2(sum_exp) / log2(e) = lg2(sum_exp) * ln(2)
    lg2.approx.f32 %f10, %f2;            // log2(sum_exp)
    mul.f32        %f11, %f10, {LN2};    // * ln(2) = ln(sum_exp)

    add.f32        %f12, %f9, %f11;      // loss = -(diag - max) + ln(sum_exp)

    // Store loss[row]
    mul.wide.u32  %rd4, %r4, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f12;

    // Grid stride
    mov.u32       %r1, %ntid.x;
    mov.u32       %r9, %nctaid.x;
    mul.lo.u32    %r9, %r1, %r9;
    add.u32       %r4, %r4, %r9;
    bra           $CL_OUTER;

$CL_DONE:
    ret;
}}
"#,
        NEG_INF = neg_inf,
        ZERO = zero,
        LOG2E = log2e,
        LN2 = ln2,
    )
}

// ─── Kernel 4: roi_align ─────────────────────────────────────────────────────

/// Per-bin bilinear RoI Align feature extraction (Faster R-CNN / Mask R-CNN).
///
/// For each output `(roi_idx, c, ph, pw)` quad, the kernel:
/// 1. Recovers the RoI box from `p_rois[roi_idx, :] = [x1, y1, x2, y2]`
/// 2. Computes the `sampling_ratio × sampling_ratio` sample point grid within the bin
/// 3. Bilinearly interpolates the feature map at each sample point
/// 4. Averages all `sampling_ratio²` samples as the bin output (using `rcp.approx.f32`)
///
/// Thread id is `roi_idx * n_chans * pooled_h * pooled_w + c * pooled_h * pooled_w + ph * pooled_w + pw`.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_feat` | `u64` (→ `f32*`) | Feature map `[n_chans × feat_h × feat_w]` |
/// | `p_rois` | `u64` (→ `f32*`) | RoI boxes `[n_rois × 4]` (xyxy, feature-map coords) |
/// | `p_out` | `u64` (→ `f32*`) | Output `[n_rois × n_chans × pooled_h × pooled_w]` |
/// | `n_rois` | `u32` | Number of RoIs |
/// | `feat_h` | `u32` | Feature map height |
/// | `feat_w` | `u32` | Feature map width |
/// | `n_chans` | `u32` | Number of channels |
/// | `pooled_h` | `u32` | Pooled output height |
/// | `pooled_w` | `u32` | Pooled output width |
/// | `sampling_ratio` | `u32` | Sampling points per bin side (total = ratio²) |
///
/// Launch: `grid = ceil(n_rois * n_chans * pooled_h * pooled_w / 256)`, `block = 256`.
#[must_use]
pub fn roi_align_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let half = f32_hex(0.5_f32);
    format!(
        r#"{hdr}.visible .entry roi_align(
    .param .u64 p_feat,
    .param .u64 p_rois,
    .param .u64 p_out,
    .param .u32 n_rois,
    .param .u32 feat_h,
    .param .u32 feat_w,
    .param .u32 n_chans,
    .param .u32 pooled_h,
    .param .u32 pooled_w,
    .param .u32 sampling_ratio
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<40>;
    .reg .f32  %f<48>;
    .reg .pred %p0, %p1, %p2;

    ld.param.u64  %rd0, [p_feat];
    ld.param.u64  %rd1, [p_rois];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n_rois];
    ld.param.u32  %r1,  [feat_h];
    ld.param.u32  %r2,  [feat_w];
    ld.param.u32  %r3,  [n_chans];
    ld.param.u32  %r4,  [pooled_h];
    ld.param.u32  %r5,  [pooled_w];
    ld.param.u32  %r6,  [sampling_ratio];

    // Grid-stride: one thread per (roi, c, ph, pw)
    // flat_tid = roi * n_chans * pooled_h * pooled_w + c * pooled_h * pooled_w + ph * pooled_w + pw
    mov.u32       %r7, %ntid.x;
    mov.u32       %r8, %ctaid.x;
    mov.u32       %r9, %tid.x;
    mad.lo.u32    %r10, %r7, %r8, %r9;   // r10 = flat_tid

    // total = n_rois * n_chans * pooled_h * pooled_w
    mul.lo.u32    %r11, %r0, %r3;
    mul.lo.u32    %r11, %r11, %r4;
    mul.lo.u32    %r11, %r11, %r5;

$RA_OUTER:
    setp.ge.u32   %p0, %r10, %r11;
    @%p0 bra $RA_DONE;

    // Decode indices
    mul.lo.u32    %r12, %r4, %r5;         // pooled_hw = pooled_h * pooled_w
    mul.lo.u32    %r13, %r3, %r12;        // chan_pool = n_chans * pooled_hw

    div.u32       %r14, %r10, %r13;       // r14 = roi_idx
    rem.u32       %r15, %r10, %r13;       // remainder in [c, ph, pw]
    div.u32       %r16, %r15, %r12;       // r16 = c
    rem.u32       %r17, %r15, %r12;       // remainder in [ph, pw]
    div.u32       %r18, %r17, %r5;        // r18 = ph
    rem.u32       %r19, %r17, %r5;        // r19 = pw

    // Load RoI box: p_rois[roi_idx, :] = [x1, y1, x2, y2]
    mul.lo.u32    %r20, %r14, 4;           // 4 floats per roi
    mul.wide.u32  %rd3, %r20, 4;           // byte offset
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];            // f0 = x1
    add.u64       %rd5, %rd4, 4;
    ld.global.f32 %f1, [%rd5];            // f1 = y1
    add.u64       %rd6, %rd5, 4;
    ld.global.f32 %f2, [%rd6];            // f2 = x2
    add.u64       %rd7, %rd6, 4;
    ld.global.f32 %f3, [%rd7];            // f3 = y2

    // RoI width and height in feature-map space
    sub.f32       %f4, %f2, %f0;          // f4 = roi_w = x2 - x1
    sub.f32       %f5, %f3, %f1;          // f5 = roi_h = y2 - y1

    // Bin width and height
    cvt.rn.f32.u32 %f6, %r5;             // (f32)pooled_w
    cvt.rn.f32.u32 %f7, %r4;             // (f32)pooled_h
    div.rn.f32    %f8, %f4, %f6;          // f8 = bin_w = roi_w / pooled_w
    div.rn.f32    %f9, %f5, %f7;          // f9 = bin_h = roi_h / pooled_h

    // Bin top-left corner (float)
    cvt.rn.f32.u32 %f10, %r18;            // (f32)ph
    cvt.rn.f32.u32 %f11, %r19;            // (f32)pw
    fma.rn.f32    %f12, %f10, %f9, %f1;   // f12 = bin_y0 = y1 + ph * bin_h
    fma.rn.f32    %f13, %f11, %f8, %f0;   // f13 = bin_x0 = x1 + pw * bin_w

    // Sample step size within bin
    cvt.rn.f32.u32 %f14, %r6;            // (f32)sampling_ratio
    div.rn.f32    %f15, %f9, %f14;        // f15 = step_y = bin_h / ratio
    div.rn.f32    %f16, %f8, %f14;        // f16 = step_x = bin_w / ratio

    // Accumulator
    mov.f32       %f17, {ZERO};           // acc = 0

    // Inner double loop: iy in [0, ratio), ix in [0, ratio)
    mov.u32       %r21, 0;                // iy = 0

$RA_IY_LOOP:
    setp.ge.u32   %p1, %r21, %r6;
    @%p1 bra $RA_IY_END;

    mov.u32       %r22, 0;                // ix = 0

$RA_IX_LOOP:
    setp.ge.u32   %p1, %r22, %r6;
    @%p1 bra $RA_IX_END;

    // Sample coordinates: sy = bin_y0 + (iy + 0.5) * step_y
    //                     sx = bin_x0 + (ix + 0.5) * step_x
    cvt.rn.f32.u32 %f18, %r21;           // (f32)iy
    fma.rn.f32    %f18, %f18, {ONE}, {HALF}; // iy + 0.5
    fma.rn.f32    %f19, %f18, %f15, %f12;    // sy = bin_y0 + (iy+0.5)*step_y

    cvt.rn.f32.u32 %f20, %r22;           // (f32)ix
    fma.rn.f32    %f20, %f20, {ONE}, {HALF}; // ix + 0.5
    fma.rn.f32    %f21, %f20, %f16, %f13;    // sx = bin_x0 + (ix+0.5)*step_x

    // Clamp sy to [0, feat_h - 1], sx to [0, feat_w - 1]
    max.f32       %f19, %f19, {ZERO};
    cvt.rn.f32.u32 %f22, %r1;
    sub.f32       %f22, %f22, {ONE};
    min.f32       %f19, %f19, %f22;

    max.f32       %f21, %f21, {ZERO};
    cvt.rn.f32.u32 %f23, %r2;
    sub.f32       %f23, %f23, {ONE};
    min.f32       %f21, %f21, %f23;

    // Bilinear interpolation at (sy, sx) in feature map channel c
    floor.f32     %f24, %f19;             // y0f
    floor.f32     %f25, %f21;             // x0f
    sub.f32       %f26, %f19, %f24;       // fy
    sub.f32       %f27, %f21, %f25;       // fx

    cvt.rzi.u32.f32 %r23, %f24;          // y0
    cvt.rzi.u32.f32 %r24, %f25;          // x0
    add.u32       %r25, %r23, 1;
    sub.u32       %r26, %r1, 1;
    min.u32       %r25, %r25, %r26;       // y1 = min(y0+1, feat_h-1)
    add.u32       %r27, %r24, 1;
    sub.u32       %r28, %r2, 1;
    min.u32       %r27, %r27, %r28;       // x1 = min(x0+1, feat_w-1)

    // Channel base offset for feature map: c * feat_h * feat_w
    mul.lo.u32    %r29, %r16, %r1;
    mul.lo.u32    %r29, %r29, %r2;

    // tl = feat[c, y0, x0]
    mad.lo.u32    %r30, %r23, %r2, %r24;
    add.u32       %r30, %r30, %r29;
    mul.wide.u32  %rd8, %r30, 4;
    add.u64       %rd9, %rd0, %rd8;
    ld.global.f32 %f28, [%rd9];

    // tr = feat[c, y0, x1]
    mad.lo.u32    %r30, %r23, %r2, %r27;
    add.u32       %r30, %r30, %r29;
    mul.wide.u32  %rd8, %r30, 4;
    add.u64       %rd9, %rd0, %rd8;
    ld.global.f32 %f29, [%rd9];

    // bl = feat[c, y1, x0]
    mad.lo.u32    %r30, %r25, %r2, %r24;
    add.u32       %r30, %r30, %r29;
    mul.wide.u32  %rd8, %r30, 4;
    add.u64       %rd9, %rd0, %rd8;
    ld.global.f32 %f30, [%rd9];

    // br = feat[c, y1, x1]
    mad.lo.u32    %r30, %r25, %r2, %r27;
    add.u32       %r30, %r30, %r29;
    mul.wide.u32  %rd8, %r30, 4;
    add.u64       %rd9, %rd0, %rd8;
    ld.global.f32 %f31, [%rd9];

    // Blend
    sub.f32       %f32, {ONE}, %f27;      // 1 - fx
    sub.f32       %f33, {ONE}, %f26;      // 1 - fy
    mul.f32       %f34, %f28, %f32;
    fma.rn.f32    %f34, %f29, %f27, %f34; // top = tl*(1-fx)+tr*fx
    mul.f32       %f35, %f30, %f32;
    fma.rn.f32    %f35, %f31, %f27, %f35; // bot = bl*(1-fx)+br*fx
    mul.f32       %f36, %f34, %f33;
    fma.rn.f32    %f36, %f35, %f26, %f36; // interp = top*(1-fy)+bot*fy

    add.f32       %f17, %f17, %f36;       // acc += interp

    add.u32       %r22, %r22, 1;
    bra           $RA_IX_LOOP;

$RA_IX_END:
    add.u32       %r21, %r21, 1;
    bra           $RA_IY_LOOP;

$RA_IY_END:
    // Divide by sampling_ratio^2 using rcp.approx
    mul.lo.u32    %r31, %r6, %r6;         // ratio^2
    cvt.rn.f32.u32 %f37, %r31;
    rcp.approx.f32 %f38, %f37;
    mul.f32       %f17, %f17, %f38;       // acc / ratio^2

    // Store output
    mul.wide.u32  %rd10, %r10, 4;
    add.u64       %rd11, %rd2, %rd10;
    st.global.f32 [%rd11], %f17;

    // Grid stride
    mov.u32       %r7, %ntid.x;
    mov.u32       %r32, %nctaid.x;
    mul.lo.u32    %r32, %r7, %r32;
    add.u32       %r10, %r10, %r32;
    bra           $RA_OUTER;

$RA_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        HALF = half,
    )
}

// ─── Kernel 5: image_normalize ───────────────────────────────────────────────

/// Channel-wise in-place normalization: `x ← (x − mean[c]) / std[c]`.
///
/// The channel index is recovered from the flat element index:
/// `c = flat_tid / (h * w)`.  The `mean[c]` and `std[c]` values
/// are loaded via indirect addressing from the `p_mean` / `p_std` arrays.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_img` | `u64` (→ `f32*`) | Image `[n_chans × h × w]` (in-place) |
/// | `p_mean` | `u64` (→ `f32*`) | Per-channel mean `[n_chans]` |
/// | `p_std` | `u64` (→ `f32*`) | Per-channel std `[n_chans]` |
/// | `h` | `u32` | Image height |
/// | `w` | `u32` | Image width |
/// | `n_chans` | `u32` | Number of channels |
///
/// Launch: `grid = ceil(n_chans * h * w / 256)`, `block = 256`.
#[must_use]
pub fn image_normalize_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry image_normalize(
    .param .u64 p_img,
    .param .u64 p_mean,
    .param .u64 p_std,
    .param .u32 h,
    .param .u32 w,
    .param .u32 n_chans
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_img];
    ld.param.u64  %rd1, [p_mean];
    ld.param.u64  %rd2, [p_std];
    ld.param.u32  %r0,  [h];
    ld.param.u32  %r1,  [w];
    ld.param.u32  %r2,  [n_chans];

    // Grid-stride: one thread per element
    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;    // r6 = flat element index

    // total = n_chans * h * w
    mul.lo.u32    %r7, %r2, %r0;
    mul.lo.u32    %r7, %r7, %r1;

$IN_OUTER:
    setp.ge.u32   %p0, %r6, %r7;
    @%p0 bra $IN_DONE;

    // c = flat_idx / (h * w)
    mul.lo.u32    %r8, %r0, %r1;          // r8 = hw = h * w
    div.u32       %r9, %r6, %r8;          // r9 = c

    // Load mean[c] and std[c]
    mul.wide.u32  %rd3, %r9, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];            // f0 = mean[c]
    add.u64       %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd5];            // f1 = std[c]

    // Load pixel value
    mul.wide.u32  %rd6, %r6, 4;
    add.u64       %rd7, %rd0, %rd6;
    ld.global.f32 %f2, [%rd7];            // f2 = x

    // rcp(std) and (x - mean) * rcp(std)
    rcp.approx.f32 %f3, %f1;             // f3 = rcp(std)
    sub.f32       %f4, %f2, %f0;          // f4 = x - mean
    mul.f32       %f5, %f4, %f3;          // f5 = (x - mean) * rcp(std)

    // Store in-place
    st.global.f32 [%rd7], %f5;

    // Grid stride
    mov.u32       %r3, %ntid.x;
    mov.u32       %r10, %nctaid.x;
    mul.lo.u32    %r10, %r3, %r10;
    add.u32       %r6, %r6, %r10;
    bra           $IN_OUTER;

$IN_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 6: adaptive_avg_pool ─────────────────────────────────────────────

/// Adaptive 2D average pooling: variable input size → fixed output size.
///
/// For each output element `(c, oh, ow)`:
/// - `h_start = floor(oh * in_h / out_h)`
/// - `h_end   = ceil((oh+1) * in_h / out_h)`
/// - `w_start = floor(ow * in_w / out_w)`
/// - `w_end   = ceil((ow+1) * in_w / out_w)`
/// - output = average of `in[c, h_start:h_end, w_start:w_end]`
///
/// Uses integer division for window bounds and `rcp.approx.f32` for the
/// divisor.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_in` | `u64` (→ `f32*`) | Input `[n_chans × in_h × in_w]` |
/// | `p_out` | `u64` (→ `f32*`) | Output `[n_chans × out_h × out_w]` |
/// | `in_h` | `u32` | Input height |
/// | `in_w` | `u32` | Input width |
/// | `out_h` | `u32` | Output height |
/// | `out_w` | `u32` | Output width |
/// | `n_chans` | `u32` | Number of channels |
///
/// Launch: `grid = ceil(n_chans * out_h * out_w / 256)`, `block = 256`.
#[must_use]
pub fn adaptive_avg_pool_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}.visible .entry adaptive_avg_pool(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .u32 in_h,
    .param .u32 in_w,
    .param .u32 out_h,
    .param .u32 out_w,
    .param .u32 n_chans
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<32>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_in];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [in_h];
    ld.param.u32  %r1,  [in_w];
    ld.param.u32  %r2,  [out_h];
    ld.param.u32  %r3,  [out_w];
    ld.param.u32  %r4,  [n_chans];

    // Grid-stride: one thread per (c, oh, ow)
    mov.u32       %r5, %ntid.x;
    mov.u32       %r6, %ctaid.x;
    mov.u32       %r7, %tid.x;
    mad.lo.u32    %r8, %r5, %r6, %r7;    // r8 = flat_tid

    // total = n_chans * out_h * out_w
    mul.lo.u32    %r9, %r4, %r2;
    mul.lo.u32    %r9, %r9, %r3;

$AAP_OUTER:
    setp.ge.u32   %p0, %r8, %r9;
    @%p0 bra $AAP_DONE;

    // Decode: c, oh, ow
    mul.lo.u32    %r10, %r2, %r3;         // out_hw
    div.u32       %r11, %r8, %r10;        // r11 = c
    rem.u32       %r12, %r8, %r10;
    div.u32       %r13, %r12, %r3;        // r13 = oh
    rem.u32       %r14, %r12, %r3;        // r14 = ow

    // h_start = floor(oh * in_h / out_h)  = (oh * in_h) / out_h
    mul.lo.u32    %r15, %r13, %r0;
    div.u32       %r15, %r15, %r2;        // r15 = h_start

    // h_end = ceil((oh+1) * in_h / out_h) = ((oh+1) * in_h + out_h - 1) / out_h
    add.u32       %r16, %r13, 1;
    mul.lo.u32    %r16, %r16, %r0;
    add.u32       %r16, %r16, %r2;
    sub.u32       %r16, %r16, 1;
    div.u32       %r16, %r16, %r2;        // r16 = h_end

    // w_start = (ow * in_w) / out_w
    mul.lo.u32    %r17, %r14, %r1;
    div.u32       %r17, %r17, %r3;        // r17 = w_start

    // w_end = ((ow+1) * in_w + out_w - 1) / out_w
    add.u32       %r18, %r14, 1;
    mul.lo.u32    %r18, %r18, %r1;
    add.u32       %r18, %r18, %r3;
    sub.u32       %r18, %r18, 1;
    div.u32       %r18, %r18, %r3;        // r18 = w_end

    // Window element count
    sub.u32       %r19, %r16, %r15;       // h_count = h_end - h_start
    sub.u32       %r20, %r18, %r17;       // w_count = w_end - w_start
    mul.lo.u32    %r21, %r19, %r20;       // n_elems = h_count * w_count

    // Channel base in input: c * in_h * in_w
    mul.lo.u32    %r22, %r11, %r0;
    mul.lo.u32    %r22, %r22, %r1;

    // Accumulate sum over window
    mov.f32       %f0, {ZERO};
    mov.u32       %r23, %r15;             // ih = h_start

$AAP_HLOOP:
    setp.ge.u32   %p1, %r23, %r16;
    @%p1 bra $AAP_HEND;

    mov.u32       %r24, %r17;             // iw = w_start

$AAP_WLOOP:
    setp.ge.u32   %p1, %r24, %r18;
    @%p1 bra $AAP_WEND;

    mad.lo.u32    %r25, %r23, %r1, %r24;  // ih * in_w + iw
    add.u32       %r25, %r25, %r22;       // + channel base
    mul.wide.u32  %rd2, %r25, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];
    add.f32       %f0, %f0, %f1;

    add.u32       %r24, %r24, 1;
    bra           $AAP_WLOOP;

$AAP_WEND:
    add.u32       %r23, %r23, 1;
    bra           $AAP_HLOOP;

$AAP_HEND:
    // Divide by n_elems
    cvt.rn.f32.u32 %f2, %r21;
    rcp.approx.f32 %f3, %f2;
    mul.f32        %f4, %f0, %f3;

    // Store
    mul.wide.u32  %rd4, %r8, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f4;

    // Grid stride
    mov.u32       %r5, %ntid.x;
    mov.u32       %r26, %nctaid.x;
    mul.lo.u32    %r26, %r5, %r26;
    add.u32       %r8, %r8, %r26;
    bra           $AAP_OUTER;

$AAP_DONE:
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 7: focal_loss ─────────────────────────────────────────────────────

/// Focal loss for dense object detection (RetinaNet).
///
/// For each `(batch, class)` pair:
/// - Sigmoid activation: `p = 1 / (1 + exp(−logit))` via `ex2.approx.f32`
/// - Positive (label = 1): `loss = −α·(1−p)^γ·log(p)`
/// - Negative (label = 0): `loss = −(1−α)·p^γ·log(1−p)`
///
/// Uses concrete constants `α = 0.25`, `γ = 2.0` embedded as `f32_hex` literals.
/// The `log(p)` is computed via `lg2.approx.f32` + multiplication by `ln(2)`.
/// The `p^γ` for γ=2 reduces to a plain squaring for efficiency.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_logits` | `u64` (→ `f32*`) | Logits `[n_elem]` (B×C flattened) |
/// | `p_labels` | `u64` (→ `f32*`) | Binary labels `[n_elem]` (0.0 or 1.0) |
/// | `p_loss` | `u64` (→ `f32*`) | Per-element loss output `[n_elem]` |
/// | `n_elem` | `u32` | Total number of elements (B×C) |
///
/// Launch: `grid = ceil(n_elem / 256)`, `block = 256`.
#[must_use]
pub fn focal_loss_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    // Concrete hyperparameters embedded in PTX
    let alpha = f32_hex(0.25_f32); // α = 0.25
    let one_minus_alpha = f32_hex(0.75_f32); // 1 − α = 0.75
    let one = f32_hex(1.0_f32);
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    let ln2 = f32_hex(std::f32::consts::LN_2);
    // Small epsilon to avoid log(0)
    let eps = f32_hex(1e-7_f32);
    // Threshold for label: 0.5
    let half = f32_hex(0.5_f32);
    format!(
        r#"{hdr}.visible .entry focal_loss(
    .param .u64 p_logits,
    .param .u64 p_labels,
    .param .u64 p_loss,
    .param .u32 n_elem
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<8>;
    .reg .f32  %f<32>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_logits];
    ld.param.u64  %rd1, [p_labels];
    ld.param.u64  %rd2, [p_loss];
    ld.param.u32  %r0,  [n_elem];

    // Grid-stride
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;    // r4 = global tid

$FL_OUTER:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $FL_DONE;

    mul.wide.u32  %rd3, %r4, 4;

    // Load logit and label
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];            // f0 = logit
    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f1, [%rd5];            // f1 = label (0.0 or 1.0)

    // ── Sigmoid: p = 1 / (1 + exp(-logit)) ──────────────────────────────────
    // exp(-logit) = ex2(-logit * log2e)
    neg.f32       %f2, %f0;               // -logit
    mul.f32       %f3, %f2, {LOG2E};
    ex2.approx.f32 %f4, %f3;             // f4 = exp(-logit)
    add.f32       %f5, %f4, {ONE};        // 1 + exp(-logit)
    rcp.approx.f32 %f6, %f5;             // f6 = p = sigmoid(logit)

    // ── log(p) via lg2(p) * ln2 ──────────────────────────────────────────────
    // clamp p to [eps, 1-eps] for numerical safety
    max.f32       %f7, %f6, {EPS};
    sub.f32       %f8, {ONE}, {EPS};
    min.f32       %f7, %f7, %f8;          // f7 = p_clamped

    lg2.approx.f32 %f9, %f7;             // log2(p)
    mul.f32        %f10, %f9, {LN2};     // f10 = ln(p)

    // ── log(1-p) ──────────────────────────────────────────────────────────────
    sub.f32       %f11, {ONE}, %f7;       // 1 - p
    max.f32       %f11, %f11, {EPS};
    lg2.approx.f32 %f12, %f11;
    mul.f32        %f13, %f12, {LN2};    // f13 = ln(1-p)

    // ── (1-p)^2 and p^2 ──────────────────────────────────────────────────────
    // γ = 2.0, so p^γ = p^2
    sub.f32       %f14, {ONE}, %f7;       // 1 - p
    mul.f32       %f15, %f14, %f14;       // f15 = (1-p)^2

    mul.f32       %f16, %f7, %f7;         // f16 = p^2

    // ── Positive branch: loss_pos = -α * (1-p)^2 * ln(p) ────────────────────
    mul.f32       %f17, %f15, %f10;       // (1-p)^2 * ln(p)
    mul.f32       %f18, {ALPHA}, %f17;    // α * ...
    neg.f32       %f19, %f18;             // loss_pos = -α*(1-p)^2*ln(p)

    // ── Negative branch: loss_neg = -(1-α) * p^2 * ln(1-p) ──────────────────
    mul.f32       %f20, %f16, %f13;       // p^2 * ln(1-p)
    mul.f32       %f21, {ONE_MINUS_ALPHA}, %f20;
    neg.f32       %f22, %f21;             // loss_neg = -(1-α)*p^2*ln(1-p)

    // ── Select branch based on label ─────────────────────────────────────────
    // label >= 0.5 → positive
    setp.ge.f32   %p1, %f1, {HALF};
    selp.f32      %f23, %f19, %f22, %p1; // f23 = chosen loss

    // Store
    add.u64       %rd6, %rd2, %rd3;
    st.global.f32 [%rd6], %f23;

    // Grid stride
    mov.u32       %r1, %ntid.x;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r5, %r1, %r5;
    add.u32       %r4, %r4, %r5;
    bra           $FL_OUTER;

$FL_DONE:
    ret;
}}
"#,
        ALPHA = alpha,
        ONE_MINUS_ALPHA = one_minus_alpha,
        ONE = one,
        LOG2E = log2e,
        LN2 = ln2,
        EPS = eps,
        HALF = half,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── f32_hex helper ────────────────────────────────────────────────────────

    #[test]
    fn f32_hex_one() {
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }

    #[test]
    fn f32_hex_zero() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
    }

    #[test]
    fn f32_hex_negative_one() {
        // -1.0f32 has bit pattern 0xBF800000
        assert_eq!(f32_hex(-1.0_f32), "0FBF800000");
    }

    #[test]
    fn f32_hex_half() {
        // 0.5f32 = 0x3F000000
        assert_eq!(f32_hex(0.5_f32), "0F3F000000");
    }

    #[test]
    fn f32_hex_two() {
        // 2.0f32 = 0x40000000
        assert_eq!(f32_hex(2.0_f32), "0F40000000");
    }

    #[test]
    fn f32_hex_neg_inf() {
        // -inf = 0xFF800000
        assert_eq!(f32_hex(f32::NEG_INFINITY), "0FFF800000");
    }

    #[test]
    fn f32_hex_pos_inf() {
        // +inf = 0x7F800000
        assert_eq!(f32_hex(f32::INFINITY), "0F7F800000");
    }

    // ── ptx_header consistency ────────────────────────────────────────────────

    #[test]
    fn ptx_header_sm75() {
        let h = ptx_header(75);
        assert!(h.contains(".version 7.5"), "sm75: {h}");
        assert!(h.contains(".target sm_75"), "sm75: {h}");
    }

    #[test]
    fn ptx_header_sm80() {
        let h = ptx_header(80);
        assert!(h.contains(".version 8.0"), "sm80: {h}");
        assert!(h.contains(".target sm_80"), "sm80: {h}");
    }

    #[test]
    fn ptx_header_sm86() {
        let h = ptx_header(86);
        assert!(h.contains(".version 8.0"), "sm86: {h}");
        assert!(h.contains(".target sm_86"), "sm86: {h}");
    }

    #[test]
    fn ptx_header_sm90() {
        let h = ptx_header(90);
        assert!(h.contains(".version 8.4"), "sm90: {h}");
        assert!(h.contains(".target sm_90"), "sm90: {h}");
    }

    #[test]
    fn ptx_header_sm100() {
        let h = ptx_header(100);
        assert!(h.contains(".version 8.7"), "sm100: {h}");
        assert!(h.contains(".target sm_100"), "sm100: {h}");
    }

    #[test]
    fn ptx_header_sm120() {
        let h = ptx_header(120);
        assert!(h.contains(".version 8.7"), "sm120: {h}");
        assert!(h.contains(".target sm_120"), "sm120: {h}");
    }

    // ── patch_embed_ptx ───────────────────────────────────────────────────────

    #[test]
    fn patch_embed_contains_target_sm80() {
        let ptx = patch_embed_ptx(80);
        assert!(ptx.contains(".target sm_80"), "missing sm_80 target");
    }

    #[test]
    fn patch_embed_contains_entry() {
        let ptx = patch_embed_ptx(80);
        assert!(ptx.contains(".visible .entry patch_embed"), "missing entry");
    }

    #[test]
    fn patch_embed_sm_versions() {
        for sm in [75u32, 80, 86, 90, 100, 120] {
            let ptx = patch_embed_ptx(sm);
            assert!(
                ptx.contains(&format!(".target sm_{sm}")),
                "patch_embed missing .target sm_{sm}"
            );
        }
    }

    #[test]
    fn patch_embed_has_fma() {
        let ptx = patch_embed_ptx(80);
        assert!(ptx.contains("fma.rn.f32"), "must use fma.rn.f32");
    }

    #[test]
    fn patch_embed_has_params() {
        let ptx = patch_embed_ptx(80);
        assert!(ptx.contains("p_in"), "missing p_in param");
        assert!(ptx.contains("p_kernel"), "missing p_kernel param");
        assert!(ptx.contains("p_bias"), "missing p_bias param");
        assert!(ptx.contains("p_out"), "missing p_out param");
        assert!(ptx.contains("n_patches"), "missing n_patches param");
        assert!(ptx.contains("embed_dim"), "missing embed_dim param");
    }

    // ── bilinear_interp_ptx ───────────────────────────────────────────────────

    #[test]
    fn bilinear_interp_contains_target_sm80() {
        let ptx = bilinear_interp_ptx(80);
        assert!(ptx.contains(".target sm_80"), "missing sm_80 target");
    }

    #[test]
    fn bilinear_interp_sm_versions() {
        for sm in [75u32, 80, 86, 90, 100, 120] {
            let ptx = bilinear_interp_ptx(sm);
            assert!(
                ptx.contains(&format!(".target sm_{sm}")),
                "bilinear_interp missing .target sm_{sm}"
            );
        }
    }

    #[test]
    fn bilinear_interp_has_floor() {
        let ptx = bilinear_interp_ptx(80);
        assert!(ptx.contains("floor.f32"), "must use floor.f32");
    }

    #[test]
    fn bilinear_interp_has_fma() {
        let ptx = bilinear_interp_ptx(80);
        assert!(ptx.contains("fma.rn.f32"), "must use fma.rn.f32");
    }

    #[test]
    fn bilinear_interp_has_params() {
        let ptx = bilinear_interp_ptx(80);
        assert!(ptx.contains("in_h"), "missing in_h param");
        assert!(ptx.contains("in_w"), "missing in_w param");
        assert!(ptx.contains("out_h"), "missing out_h param");
        assert!(ptx.contains("out_w"), "missing out_w param");
        assert!(ptx.contains("n_chans"), "missing n_chans param");
    }

    // ── contrastive_loss_ptx ──────────────────────────────────────────────────

    #[test]
    fn contrastive_loss_contains_target_sm80() {
        let ptx = contrastive_loss_ptx(80);
        assert!(ptx.contains(".target sm_80"), "missing sm_80 target");
    }

    #[test]
    fn contrastive_loss_sm_versions() {
        for sm in [75u32, 80, 86, 90, 100, 120] {
            let ptx = contrastive_loss_ptx(sm);
            assert!(
                ptx.contains(&format!(".target sm_{sm}")),
                "contrastive_loss missing .target sm_{sm}"
            );
        }
    }

    #[test]
    fn contrastive_loss_has_ex2() {
        let ptx = contrastive_loss_ptx(80);
        assert!(ptx.contains("ex2.approx.f32"), "must use ex2.approx.f32");
    }

    #[test]
    fn contrastive_loss_has_lg2() {
        let ptx = contrastive_loss_ptx(80);
        assert!(ptx.contains("lg2.approx.f32"), "must use lg2.approx.f32");
    }

    #[test]
    fn contrastive_loss_has_neg_inf() {
        let ptx = contrastive_loss_ptx(80);
        // NEG_INF is used in max reduction initialisation
        assert!(ptx.contains("FF800000"), "must contain NEG_INF constant");
    }

    #[test]
    fn contrastive_loss_three_pass_labels() {
        let ptx = contrastive_loss_ptx(80);
        // Three distinct loop labels
        assert!(ptx.contains("$CL_MAX_LOOP"), "missing max loop label");
        assert!(ptx.contains("$CL_SUM_LOOP"), "missing sum loop label");
        assert!(ptx.contains("p_sim"), "missing p_sim param");
        assert!(ptx.contains("p_loss"), "missing p_loss param");
        assert!(ptx.contains("n_batch"), "missing n_batch param");
    }

    // ── roi_align_ptx ─────────────────────────────────────────────────────────

    #[test]
    fn roi_align_contains_target_sm80() {
        let ptx = roi_align_ptx(80);
        assert!(ptx.contains(".target sm_80"), "missing sm_80 target");
    }

    #[test]
    fn roi_align_sm_versions() {
        for sm in [75u32, 80, 86, 90, 100, 120] {
            let ptx = roi_align_ptx(sm);
            assert!(
                ptx.contains(&format!(".target sm_{sm}")),
                "roi_align missing .target sm_{sm}"
            );
        }
    }

    #[test]
    fn roi_align_has_rcp_approx() {
        let ptx = roi_align_ptx(80);
        assert!(ptx.contains("rcp.approx.f32"), "must use rcp.approx.f32");
    }

    #[test]
    fn roi_align_has_fma() {
        let ptx = roi_align_ptx(80);
        assert!(ptx.contains("fma.rn.f32"), "must use fma.rn.f32");
    }

    #[test]
    fn roi_align_has_params() {
        let ptx = roi_align_ptx(80);
        assert!(ptx.contains("p_feat"), "missing p_feat");
        assert!(ptx.contains("p_rois"), "missing p_rois");
        assert!(ptx.contains("pooled_h"), "missing pooled_h");
        assert!(ptx.contains("pooled_w"), "missing pooled_w");
        assert!(ptx.contains("sampling_ratio"), "missing sampling_ratio");
        assert!(ptx.contains("feat_h"), "missing feat_h");
        assert!(ptx.contains("feat_w"), "missing feat_w");
    }

    #[test]
    fn roi_align_nested_loops() {
        let ptx = roi_align_ptx(80);
        assert!(ptx.contains("$RA_IY_LOOP"), "missing iy loop");
        assert!(ptx.contains("$RA_IX_LOOP"), "missing ix loop");
    }

    // ── image_normalize_ptx ───────────────────────────────────────────────────

    #[test]
    fn image_normalize_contains_target_sm80() {
        let ptx = image_normalize_ptx(80);
        assert!(ptx.contains(".target sm_80"), "missing sm_80 target");
    }

    #[test]
    fn image_normalize_sm_versions() {
        for sm in [75u32, 80, 86, 90, 100, 120] {
            let ptx = image_normalize_ptx(sm);
            assert!(
                ptx.contains(&format!(".target sm_{sm}")),
                "image_normalize missing .target sm_{sm}"
            );
        }
    }

    #[test]
    fn image_normalize_has_rcp() {
        let ptx = image_normalize_ptx(80);
        assert!(ptx.contains("rcp.approx.f32"), "must use rcp.approx.f32");
    }

    #[test]
    fn image_normalize_has_params() {
        let ptx = image_normalize_ptx(80);
        assert!(ptx.contains("p_img"), "missing p_img");
        assert!(ptx.contains("p_mean"), "missing p_mean");
        assert!(ptx.contains("p_std"), "missing p_std");
        assert!(ptx.contains("n_chans"), "missing n_chans");
    }

    #[test]
    fn image_normalize_in_place_label() {
        let ptx = image_normalize_ptx(80);
        // Must store back to p_img (same pointer as load)
        assert!(ptx.contains("st.global.f32"), "must store result");
        assert!(ptx.contains("ld.global.f32"), "must load input");
    }

    // ── adaptive_avg_pool_ptx ─────────────────────────────────────────────────

    #[test]
    fn adaptive_avg_pool_contains_target_sm80() {
        let ptx = adaptive_avg_pool_ptx(80);
        assert!(ptx.contains(".target sm_80"), "missing sm_80 target");
    }

    #[test]
    fn adaptive_avg_pool_sm_versions() {
        for sm in [75u32, 80, 86, 90, 100, 120] {
            let ptx = adaptive_avg_pool_ptx(sm);
            assert!(
                ptx.contains(&format!(".target sm_{sm}")),
                "adaptive_avg_pool missing .target sm_{sm}"
            );
        }
    }

    #[test]
    fn adaptive_avg_pool_has_rcp() {
        let ptx = adaptive_avg_pool_ptx(80);
        assert!(ptx.contains("rcp.approx.f32"), "must use rcp.approx.f32");
    }

    #[test]
    fn adaptive_avg_pool_has_params() {
        let ptx = adaptive_avg_pool_ptx(80);
        assert!(ptx.contains("in_h"), "missing in_h");
        assert!(ptx.contains("in_w"), "missing in_w");
        assert!(ptx.contains("out_h"), "missing out_h");
        assert!(ptx.contains("out_w"), "missing out_w");
        assert!(ptx.contains("n_chans"), "missing n_chans");
    }

    #[test]
    fn adaptive_avg_pool_nested_loops() {
        let ptx = adaptive_avg_pool_ptx(80);
        assert!(ptx.contains("$AAP_HLOOP"), "missing height loop");
        assert!(ptx.contains("$AAP_WLOOP"), "missing width loop");
    }

    #[test]
    fn adaptive_avg_pool_window_bounds_ceil_floor() {
        let ptx = adaptive_avg_pool_ptx(80);
        // Ceiling formula uses (x + out_h - 1) / out_h → multiple div.u32
        assert!(ptx.contains("div.u32"), "must use integer division");
    }

    // ── focal_loss_ptx ────────────────────────────────────────────────────────

    #[test]
    fn focal_loss_contains_target_sm80() {
        let ptx = focal_loss_ptx(80);
        assert!(ptx.contains(".target sm_80"), "missing sm_80 target");
    }

    #[test]
    fn focal_loss_sm_versions() {
        for sm in [75u32, 80, 86, 90, 100, 120] {
            let ptx = focal_loss_ptx(sm);
            assert!(
                ptx.contains(&format!(".target sm_{sm}")),
                "focal_loss missing .target sm_{sm}"
            );
        }
    }

    #[test]
    fn focal_loss_has_alpha_constant() {
        let ptx = focal_loss_ptx(80);
        // α = 0.25 = 0x3E800000
        assert!(ptx.contains("3E800000"), "must embed alpha=0.25 constant");
    }

    #[test]
    fn focal_loss_has_ex2() {
        let ptx = focal_loss_ptx(80);
        assert!(
            ptx.contains("ex2.approx.f32"),
            "must use ex2.approx.f32 for sigmoid"
        );
    }

    #[test]
    fn focal_loss_has_lg2() {
        let ptx = focal_loss_ptx(80);
        assert!(
            ptx.contains("lg2.approx.f32"),
            "must use lg2.approx.f32 for log"
        );
    }

    #[test]
    fn focal_loss_has_selp() {
        let ptx = focal_loss_ptx(80);
        assert!(
            ptx.contains("selp.f32"),
            "must use selp.f32 for branch selection"
        );
    }

    #[test]
    fn focal_loss_has_params() {
        let ptx = focal_loss_ptx(80);
        assert!(ptx.contains("p_logits"), "missing p_logits");
        assert!(ptx.contains("p_labels"), "missing p_labels");
        assert!(ptx.contains("p_loss"), "missing p_loss");
        assert!(ptx.contains("n_elem"), "missing n_elem");
    }

    #[test]
    fn focal_loss_gamma_two_squared() {
        let ptx = focal_loss_ptx(80);
        // γ=2 is implemented by squaring → mul.f32 (no pow instruction)
        assert!(ptx.contains("mul.f32"), "must use mul.f32 for p^2");
    }

    // ── Cross-kernel all-SM sweep ─────────────────────────────────────────────

    #[test]
    #[allow(clippy::type_complexity)]
    fn all_kernels_all_sm_versions_have_target() {
        let sm_versions: &[u32] = &[75, 80, 86, 90, 100, 120];
        let generators: &[(&str, fn(u32) -> String)] = &[
            ("patch_embed", patch_embed_ptx as fn(u32) -> String),
            ("bilinear_interp", bilinear_interp_ptx as fn(u32) -> String),
            (
                "contrastive_loss",
                contrastive_loss_ptx as fn(u32) -> String,
            ),
            ("roi_align", roi_align_ptx as fn(u32) -> String),
            ("image_normalize", image_normalize_ptx as fn(u32) -> String),
            (
                "adaptive_avg_pool",
                adaptive_avg_pool_ptx as fn(u32) -> String,
            ),
            ("focal_loss", focal_loss_ptx as fn(u32) -> String),
        ];
        for &sm in sm_versions {
            for (name, kern_fn) in generators {
                let ptx = kern_fn(sm);
                assert!(
                    ptx.contains(&format!(".target sm_{sm}")),
                    "kernel '{name}' sm={sm} missing .target directive"
                );
                assert!(
                    ptx.contains(".visible .entry"),
                    "kernel '{name}' sm={sm} missing .visible .entry"
                );
            }
        }
    }

    #[test]
    fn all_kernels_version_string_sm120() {
        assert!(patch_embed_ptx(120).contains(".version 8.7"));
        assert!(bilinear_interp_ptx(120).contains(".version 8.7"));
        assert!(contrastive_loss_ptx(120).contains(".version 8.7"));
        assert!(roi_align_ptx(120).contains(".version 8.7"));
        assert!(image_normalize_ptx(120).contains(".version 8.7"));
        assert!(adaptive_avg_pool_ptx(120).contains(".version 8.7"));
        assert!(focal_loss_ptx(120).contains(".version 8.7"));
    }

    #[test]
    fn all_kernels_version_string_sm90() {
        assert!(patch_embed_ptx(90).contains(".version 8.4"));
        assert!(bilinear_interp_ptx(90).contains(".version 8.4"));
        assert!(contrastive_loss_ptx(90).contains(".version 8.4"));
        assert!(roi_align_ptx(90).contains(".version 8.4"));
        assert!(image_normalize_ptx(90).contains(".version 8.4"));
        assert!(adaptive_avg_pool_ptx(90).contains(".version 8.4"));
        assert!(focal_loss_ptx(90).contains(".version 8.4"));
    }

    #[test]
    fn all_kernels_version_string_sm80() {
        assert!(patch_embed_ptx(80).contains(".version 8.0"));
        assert!(bilinear_interp_ptx(80).contains(".version 8.0"));
        assert!(contrastive_loss_ptx(80).contains(".version 8.0"));
        assert!(roi_align_ptx(80).contains(".version 8.0"));
        assert!(image_normalize_ptx(80).contains(".version 8.0"));
        assert!(adaptive_avg_pool_ptx(80).contains(".version 8.0"));
        assert!(focal_loss_ptx(80).contains(".version 8.0"));
    }

    #[test]
    fn all_kernels_version_string_sm75() {
        assert!(patch_embed_ptx(75).contains(".version 7.5"));
        assert!(bilinear_interp_ptx(75).contains(".version 7.5"));
        assert!(contrastive_loss_ptx(75).contains(".version 7.5"));
        assert!(roi_align_ptx(75).contains(".version 7.5"));
        assert!(image_normalize_ptx(75).contains(".version 7.5"));
        assert!(adaptive_avg_pool_ptx(75).contains(".version 7.5"));
        assert!(focal_loss_ptx(75).contains(".version 7.5"));
    }

    // ── Non-empty outputs ─────────────────────────────────────────────────────

    #[test]
    fn all_kernels_produce_nonempty_strings() {
        assert!(!patch_embed_ptx(80).is_empty());
        assert!(!bilinear_interp_ptx(80).is_empty());
        assert!(!contrastive_loss_ptx(80).is_empty());
        assert!(!roi_align_ptx(80).is_empty());
        assert!(!image_normalize_ptx(80).is_empty());
        assert!(!adaptive_avg_pool_ptx(80).is_empty());
        assert!(!focal_loss_ptx(80).is_empty());
    }
}
