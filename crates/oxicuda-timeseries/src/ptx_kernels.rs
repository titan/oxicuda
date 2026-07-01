//! PTX kernel generators for `oxicuda-timeseries`.
//!
//! Each function returns a PTX program string for the requested SM version.
//! SM versions: 75, 80, 86, 90, 100, 120.
//!
//! Kernels:
//! 1. `moving_average_ptx`   — strided moving average over the time axis
//! 2. `patch_embed_1d_ptx`   — extract overlapping 1-D patches
//! 3. `causal_temporal_conv_ptx` — dilated causal 1-D conv for TCN
//! 4. `auto_correlation_ptx` — FFT-based auto-correlation (Autoformer/TimesNet)
//! 5. `revin_normalize_ptx`  — reversible instance normalisation (RevIN)
//! 6. `multirate_pool_ptx`   — average pooling at a variable stride for NHiTS
//! 7. `period_detect_ptx`    — top-k FFT magnitude frequency selection (TimesNet)

// ─── Private helpers ─────────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    let (ptx_ver, sm_str) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {sm_str}\n.address_size 64\n\n")
}

fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── 1. moving_average_ptx ───────────────────────────────────────────────────

/// Moving average over the time axis of a `[N, T]` tensor.
///
/// Each thread handles one `(n, t)` output element:
/// `out[n, t] = (1/K) * Σ_{k=0}^{K-1} in[n, clamp(t - K/2 + k, 0, T-1)]`
///
/// Parameters passed as kernel args: `in_ptr`, `out_ptr`, `N`, `T`, `K`.
pub fn moving_average_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let inv_k_comment = "// inv_k = 1.0f / K, computed from register";
    format!(
        r#"{hdr}// moving_average: [N, T] -> [N, T]  kernel_size=K
// Thread (tid_x + bid_x * bdim_x) maps to flat index (n * T + t).
{inv_k_comment}
.visible .entry moving_average(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 N,
    .param .u32 T,
    .param .u32 K
)
{{
    .reg .u64 %in<2>, %out<2>;
    .reg .u32 %n, %t, %T, %K, %tidx, %bdim, %bid, %gid, %idx, %half_k;
    .reg .u32 %k, %src_t, %src_idx, %n_off;
    .reg .f32 %acc, %val, %inv_k;
    .reg .pred %p;

    ld.param.u64 %in0, [in_ptr];
    ld.param.u64 %out0, [out_ptr];
    ld.param.u32 %T, [T];
    ld.param.u32 %K, [K];

    mov.u32 %tidx, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %gid, %bid, %bdim, %tidx;

    // gid = n * T + t
    div.u32 %n, %gid, %T;
    rem.u32 %t, %gid, %T;

    // Check bounds
    .reg .u32 %N_reg;
    ld.param.u32 %N_reg, [N];
    setp.ge.u32 %p, %n, %N_reg;
    @%p bra DONE;

    // half_k = K / 2
    shr.u32 %half_k, %K, 1;

    // inv_k = 1.0 / K
    cvt.rn.f32.u32 %inv_k, %K;
    rcp.approx.f32 %inv_k, %inv_k;

    // n_off = n * T (byte offset: n * T * 4)
    mul.lo.u32 %n_off, %n, %T;

    mov.f32 %acc, {zero};
    mov.u32 %k, 0;
LOOP:
    setp.ge.u32 %p, %k, %K;
    @%p bra END_LOOP;

    // src_t = clamp(t - half_k + k, 0, T - 1)
    add.u32 %src_t, %t, %k;
    sub.u32 %src_t, %src_t, %half_k;
    // clamp low
    .reg .s32 %src_s;
    cvt.s32.u32 %src_s, %src_t;
    max.s32 %src_s, %src_s, 0;
    // clamp high
    .reg .u32 %T_m1;
    sub.u32 %T_m1, %T, 1;
    cvt.u32.s32 %src_t, %src_s;
    min.u32 %src_t, %src_t, %T_m1;

    // load in[n, src_t]
    add.u32 %src_idx, %n_off, %src_t;
    .reg .u64 %addr;
    cvt.u64.u32 %addr, %src_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in0, %addr;
    ld.global.f32 %val, [%addr];
    add.f32 %acc, %acc, %val;

    add.u32 %k, %k, 1;
    bra LOOP;
END_LOOP:
    mul.f32 %acc, %acc, %inv_k;

    // store out[n, t]
    cvt.u64.u32 %addr, %gid;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out0, %addr;
    st.global.f32 [%addr], %acc;
DONE:
    ret;
}}
"#,
        zero = f32_hex(0.0_f32)
    )
}

// ─── 2. patch_embed_1d_ptx ───────────────────────────────────────────────────

/// Extract 1-D patches from a `[N, T]` tensor.
///
/// Output shape: `[N, num_patches, patch_len]` where
/// `num_patches = (T - patch_len) / stride + 1`.
/// Each thread handles one `(n, p, l)` element.
pub fn patch_embed_1d_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// patch_embed_1d: [N, T] -> [N, num_patches, patch_len]
// args: in_ptr, out_ptr, N, T, patch_len, stride, num_patches
.visible .entry patch_embed_1d(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 N,
    .param .u32 T,
    .param .u32 patch_len,
    .param .u32 stride,
    .param .u32 num_patches
)
{{
    .reg .u64 %in<2>, %out<2>, %addr;
    .reg .u32 %gid, %n, %rem, %p_idx, %l_idx, %bdim, %bid;
    .reg .u32 %T, %pl, %stride, %np;
    .reg .u32 %t_start, %t_src, %src_flat, %out_flat;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %in0, [in_ptr];
    ld.param.u64 %out0, [out_ptr];
    ld.param.u32 %T, [T];
    ld.param.u32 %pl, [patch_len];
    ld.param.u32 %stride, [stride];
    ld.param.u32 %np, [num_patches];

    mov.u32 %gid, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %gid, %bid, %bdim, %gid;

    // total = N * num_patches * patch_len
    .reg .u32 %N_reg, %total;
    ld.param.u32 %N_reg, [N];
    mul.lo.u32 %total, %np, %pl;
    mul.lo.u32 %total, %total, %N_reg;
    setp.ge.u32 %p, %gid, %total;
    @%p bra DONE;

    // n = gid / (np * pl)
    .reg .u32 %np_pl;
    mul.lo.u32 %np_pl, %np, %pl;
    div.u32 %n, %gid, %np_pl;
    rem.u32 %rem, %gid, %np_pl;

    // p_idx = rem / pl, l_idx = rem % pl
    div.u32 %p_idx, %rem, %pl;
    rem.u32 %l_idx, %rem, %pl;

    // t_src = p_idx * stride + l_idx
    mad.lo.u32 %t_src, %p_idx, %stride, %l_idx;

    // bounds check: t_src < T
    setp.ge.u32 %p, %t_src, %T;
    @%p bra DONE;

    // src_flat = n * T + t_src
    mad.lo.u32 %src_flat, %n, %T, %t_src;
    cvt.u64.u32 %addr, %src_flat;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in0, %addr;
    ld.global.f32 %val, [%addr];

    cvt.u64.u32 %addr, %gid;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out0, %addr;
    st.global.f32 [%addr], %val;
DONE:
    ret;
}}
"#
    )
}

// ─── 3. causal_temporal_conv_ptx ─────────────────────────────────────────────

/// Dilated causal 1-D convolution for TCN residual blocks.
///
/// Input/output shape: `[N, C, T]` (channels-first).
/// Causal padding: `(K - 1) * d` zeros prepended on the left.
/// `y[n,c_out,t] = Σ_{c_in,k} w[c_out,c_in,k] * x[n,c_in,t - k*d]`
pub fn causal_temporal_conv_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// causal_temporal_conv: dilated causal 1D conv [N, C_in, T] -> [N, C_out, T]
// args: in_ptr, w_ptr, b_ptr, out_ptr, N, C_in, C_out, T, K, dilation
.visible .entry causal_temporal_conv(
    .param .u64 in_ptr,
    .param .u64 w_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 N,
    .param .u32 C_in,
    .param .u32 C_out,
    .param .u32 T,
    .param .u32 K,
    .param .u32 dilation
)
{{
    .reg .u64 %in0, %w0, %b0, %out0, %addr;
    .reg .u32 %gid, %n, %c_out, %t, %rem, %bdim, %bid;
    .reg .u32 %N, %C_in, %C_out, %T, %K, %d;
    .reg .u32 %c_in, %k, %t_src, %t_s32;
    .reg .u32 %src_idx, %w_idx;
    .reg .f32 %acc, %val, %wval, %bval;
    .reg .pred %p;
    .reg .s32 %t_signed;

    ld.param.u64 %in0, [in_ptr];
    ld.param.u64 %w0, [w_ptr];
    ld.param.u64 %b0, [b_ptr];
    ld.param.u64 %out0, [out_ptr];
    ld.param.u32 %N, [N];
    ld.param.u32 %C_in, [C_in];
    ld.param.u32 %C_out, [C_out];
    ld.param.u32 %T, [T];
    ld.param.u32 %K, [K];
    ld.param.u32 %d, [dilation];

    mov.u32 %gid, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %gid, %bid, %bdim, %gid;

    // total = N * C_out * T
    .reg .u32 %total, %ct;
    mul.lo.u32 %ct, %C_out, %T;
    mul.lo.u32 %total, %N, %ct;
    setp.ge.u32 %p, %gid, %total;
    @%p bra DONE;

    // n = gid / (C_out * T)
    div.u32 %n, %gid, %ct;
    rem.u32 %rem, %gid, %ct;
    div.u32 %c_out, %rem, %T;
    rem.u32 %t, %rem, %T;

    // load bias
    cvt.u64.u32 %addr, %c_out;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %b0, %addr;
    ld.global.f32 %bval, [%addr];
    mov.f32 %acc, %bval;

    mov.u32 %c_in, 0;
LOOP_CIN:
    setp.ge.u32 %p, %c_in, %C_in;
    @%p bra END_CIN;

    mov.u32 %k, 0;
LOOP_K:
    setp.ge.u32 %p, %k, %K;
    @%p bra END_K;

    // t_signed = t - k * dilation  (causal — skip if negative)
    mul.lo.u32 %t_src, %k, %d;
    cvt.s32.u32 %t_signed, %t;
    .reg .s32 %ks;
    cvt.s32.u32 %ks, %t_src;
    sub.s32 %t_signed, %t_signed, %ks;
    setp.lt.s32 %p, %t_signed, 0;
    @%p bra SKIP_K;

    cvt.u32.s32 %t_src, %t_signed;

    // src_idx = n * C_in * T + c_in * T + t_src
    .reg .u32 %n_off, %c_off;
    mul.lo.u32 %n_off, %n, %C_in;
    add.u32 %n_off, %n_off, %c_in;
    mul.lo.u32 %n_off, %n_off, %T;
    add.u32 %src_idx, %n_off, %t_src;
    cvt.u64.u32 %addr, %src_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in0, %addr;
    ld.global.f32 %val, [%addr];

    // w_idx = (c_out * C_in + c_in) * K + k
    .reg .u32 %wrow;
    mul.lo.u32 %wrow, %c_out, %C_in;
    add.u32 %wrow, %wrow, %c_in;
    mul.lo.u32 %wrow, %wrow, %K;
    add.u32 %w_idx, %wrow, %k;
    cvt.u64.u32 %addr, %w_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %w0, %addr;
    ld.global.f32 %wval, [%addr];
    fma.rn.f32 %acc, %val, %wval, %acc;

SKIP_K:
    add.u32 %k, %k, 1;
    bra LOOP_K;
END_K:
    add.u32 %c_in, %c_in, 1;
    bra LOOP_CIN;
END_CIN:

    cvt.u64.u32 %addr, %gid;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out0, %addr;
    st.global.f32 [%addr], %acc;
DONE:
    ret;
}}
"#
    )
}

// ─── 4. auto_correlation_ptx ─────────────────────────────────────────────────

/// FFT-based auto-correlation kernel for Autoformer / TimesNet.
///
/// Computes the circular auto-correlation of each channel in `[N, C, T]`
/// using the Wiener-Khintchine theorem: `R = IFFT(|FFT(x)|²)`.
/// This kernel is a direct PTX translation of the `|FFT|²` step
/// (the actual FFT call is done on the host via cuFFT / oxicuda-fft).
/// Here it squares the complex magnitude: `out[i] = re[i]² + im[i]²`.
pub fn auto_correlation_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// auto_correlation: |FFT(x)|^2 step for [N, C, T/2+1] complex tensor
// in_re, in_im: real/imaginary parts (packed separately for simplicity)
// out: magnitude squared [N, C, T/2+1]
.visible .entry auto_correlation(
    .param .u64 re_ptr,
    .param .u64 im_ptr,
    .param .u64 out_ptr,
    .param .u32 total
)
{{
    .reg .u64 %re0, %im0, %out0, %addr;
    .reg .u32 %gid, %bdim, %bid;
    .reg .f32 %re, %im, %mag2;
    .reg .pred %p;

    ld.param.u64 %re0, [re_ptr];
    ld.param.u64 %im0, [im_ptr];
    ld.param.u64 %out0, [out_ptr];
    ld.param.u32 %gid, [total];   // reuse reg for total

    mov.u32 %gid, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %gid, %bid, %bdim, %gid;

    .reg .u32 %tot;
    ld.param.u32 %tot, [total];
    setp.ge.u32 %p, %gid, %tot;
    @%p bra DONE;

    cvt.u64.u32 %addr, %gid;
    shl.b64 %addr, %addr, 2;

    add.u64 %addr, %re0, %addr;
    ld.global.f32 %re, [%addr];

    cvt.u64.u32 %addr, %gid;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %im0, %addr;
    ld.global.f32 %im, [%addr];

    // mag2 = re*re + im*im
    mul.f32 %mag2, %re, %re;
    fma.rn.f32 %mag2, %im, %im, %mag2;

    cvt.u64.u32 %addr, %gid;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out0, %addr;
    st.global.f32 [%addr], %mag2;
DONE:
    ret;
}}
"#
    )
}

// ─── 5. revin_normalize_ptx ──────────────────────────────────────────────────

/// Reversible Instance Normalisation (RevIN) — normalise step.
///
/// For each `(n, c)` channel in `[N, C, T]`:
/// `y[n,c,t] = (x[n,c,t] - mean_nc) / (std_nc + eps) * gamma[c] + beta[c]`
/// where `mean_nc` and `std_nc` are the per-instance per-channel statistics
/// pre-computed by the host and passed as vectors of length `N * C`.
pub fn revin_normalize_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let eps_hex = f32_hex(1e-5_f32);
    format!(
        r#"{hdr}// revin_normalize: [N, C, T] in-place with per-(n,c) stats
// args: x_ptr, mean_ptr (N*C), std_ptr (N*C), gamma_ptr (C), beta_ptr (C),
//        out_ptr, N, C, T
.visible .entry revin_normalize(
    .param .u64 x_ptr,
    .param .u64 mean_ptr,
    .param .u64 std_ptr,
    .param .u64 gamma_ptr,
    .param .u64 beta_ptr,
    .param .u64 out_ptr,
    .param .u32 N,
    .param .u32 C,
    .param .u32 T
)
{{
    .reg .u64 %x0, %mu0, %sig0, %gam0, %bet0, %out0, %addr;
    .reg .u32 %gid, %n, %c, %t, %rem, %nc_idx, %CT, %N, %C_reg, %T_reg, %bdim, %bid;
    .reg .f32 %val, %mu, %sig, %gam, %bet, %norm;
    .reg .pred %p;

    ld.param.u64 %x0,   [x_ptr];
    ld.param.u64 %mu0,  [mean_ptr];
    ld.param.u64 %sig0, [std_ptr];
    ld.param.u64 %gam0, [gamma_ptr];
    ld.param.u64 %bet0, [beta_ptr];
    ld.param.u64 %out0, [out_ptr];
    ld.param.u32 %N,    [N];
    ld.param.u32 %C_reg,[C];
    ld.param.u32 %T_reg,[T];

    mov.u32 %gid, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %gid, %bid, %bdim, %gid;

    mul.lo.u32 %CT, %C_reg, %T_reg;
    .reg .u32 %total;
    mul.lo.u32 %total, %N, %CT;
    setp.ge.u32 %p, %gid, %total;
    @%p bra DONE;

    // n, c, t
    div.u32 %n, %gid, %CT;
    rem.u32 %rem, %gid, %CT;
    div.u32 %c, %rem, %T_reg;
    rem.u32 %t, %rem, %T_reg;

    // nc_idx = n * C + c
    mad.lo.u32 %nc_idx, %n, %C_reg, %c;

    // load mean, std
    cvt.u64.u32 %addr, %nc_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %mu0, %addr;
    ld.global.f32 %mu, [%addr];

    cvt.u64.u32 %addr, %nc_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %sig0, %addr;
    ld.global.f32 %sig, [%addr];

    // load gamma, beta (per-channel)
    cvt.u64.u32 %addr, %c;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %gam0, %addr;
    ld.global.f32 %gam, [%addr];

    cvt.u64.u32 %addr, %c;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %bet0, %addr;
    ld.global.f32 %bet, [%addr];

    // load x
    cvt.u64.u32 %addr, %gid;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %x0, %addr;
    ld.global.f32 %val, [%addr];

    // norm = (val - mu) / (sig + eps) * gamma + beta
    sub.f32 %norm, %val, %mu;
    add.f32 %sig, %sig, {eps};
    div.approx.f32 %norm, %norm, %sig;
    fma.rn.f32 %norm, %norm, %gam, %bet;

    cvt.u64.u32 %addr, %gid;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out0, %addr;
    st.global.f32 [%addr], %norm;
DONE:
    ret;
}}
"#,
        eps = eps_hex
    )
}

// ─── 6. multirate_pool_ptx ───────────────────────────────────────────────────

/// Average pooling at a variable stride for NHiTS multi-rate sampling.
///
/// Input `[N, C, T]` → output `[N, C, T_out]` where `T_out = T / pool_stride`
/// (integer division; trailing elements discarded).
/// Each thread outputs one `(n, c, t_out)` element.
pub fn multirate_pool_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// multirate_pool: average pool [N, C, T] -> [N, C, T_out], T_out = T / stride
// args: in_ptr, out_ptr, N, C, T, pool_stride
.visible .entry multirate_pool(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 N,
    .param .u32 C,
    .param .u32 T,
    .param .u32 pool_stride
)
{{
    .reg .u64 %in0, %out0, %addr;
    .reg .u32 %gid, %n, %c, %t_out, %rem, %bdim, %bid;
    .reg .u32 %N, %C_reg, %T, %ps, %T_out, %CT_out;
    .reg .u32 %k, %t_src, %src_idx;
    .reg .f32 %acc, %val, %inv_ps;
    .reg .pred %p;

    ld.param.u64 %in0, [in_ptr];
    ld.param.u64 %out0, [out_ptr];
    ld.param.u32 %N, [N];
    ld.param.u32 %C_reg, [C];
    ld.param.u32 %T, [T];
    ld.param.u32 %ps, [pool_stride];

    // T_out = T / ps
    div.u32 %T_out, %T, %ps;

    mul.lo.u32 %CT_out, %C_reg, %T_out;
    .reg .u32 %total;
    mul.lo.u32 %total, %N, %CT_out;

    mov.u32 %gid, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %gid, %bid, %bdim, %gid;
    setp.ge.u32 %p, %gid, %total;
    @%p bra DONE;

    div.u32 %n, %gid, %CT_out;
    rem.u32 %rem, %gid, %CT_out;
    div.u32 %c, %rem, %T_out;
    rem.u32 %t_out, %rem, %T_out;

    // inv_ps = 1.0 / ps
    cvt.rn.f32.u32 %inv_ps, %ps;
    rcp.approx.f32 %inv_ps, %inv_ps;

    mov.f32 %acc, {zero};
    mov.u32 %k, 0;
POOL_LOOP:
    setp.ge.u32 %p, %k, %ps;
    @%p bra POOL_END;

    // t_src = t_out * ps + k
    mad.lo.u32 %t_src, %t_out, %ps, %k;

    // src_idx = n * C * T + c * T + t_src
    .reg .u32 %nct;
    mul.lo.u32 %nct, %n, %C_reg;
    add.u32 %nct, %nct, %c;
    mul.lo.u32 %nct, %nct, %T;
    add.u32 %src_idx, %nct, %t_src;

    cvt.u64.u32 %addr, %src_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in0, %addr;
    ld.global.f32 %val, [%addr];
    add.f32 %acc, %acc, %val;

    add.u32 %k, %k, 1;
    bra POOL_LOOP;
POOL_END:
    mul.f32 %acc, %acc, %inv_ps;

    cvt.u64.u32 %addr, %gid;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out0, %addr;
    st.global.f32 [%addr], %acc;
DONE:
    ret;
}}
"#,
        zero = f32_hex(0.0_f32)
    )
}

// ─── 7. period_detect_ptx ────────────────────────────────────────────────────

/// Top-K FFT magnitude frequency selection for TimesNet period detection.
///
/// Given the magnitude spectrum `[N, C, F]` (F = T/2+1 frequencies),
/// compute the average magnitude across the `(N, C)` batch and write the
/// index of the top-1 frequency (the period used in the first block).
/// A full top-k sort is done on the host; this kernel computes the
/// per-element reduction step: `avg_mag[f] = mean_{n,c}(mag[n,c,f])`.
pub fn period_detect_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// period_detect: mean magnitude across batch [N*C, F] -> [F]
// args: mag_ptr (N*C*F), out_ptr (F), NC, F
.visible .entry period_detect(
    .param .u64 mag_ptr,
    .param .u64 out_ptr,
    .param .u32 NC,
    .param .u32 F
)
{{
    .reg .u64 %mag0, %out0, %addr;
    .reg .u32 %f_idx, %nc, %src_idx, %bdim, %bid;
    .reg .u32 %NC, %F;
    .reg .f32 %acc, %val, %inv_nc;
    .reg .pred %p;

    ld.param.u64 %mag0, [mag_ptr];
    ld.param.u64 %out0, [out_ptr];
    ld.param.u32 %NC, [NC];
    ld.param.u32 %F, [F];

    mov.u32 %f_idx, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %bid, %ctaid.x;
    mad.lo.u32 %f_idx, %bid, %bdim, %f_idx;
    setp.ge.u32 %p, %f_idx, %F;
    @%p bra DONE;

    // inv_nc = 1.0 / NC
    cvt.rn.f32.u32 %inv_nc, %NC;
    rcp.approx.f32 %inv_nc, %inv_nc;

    mov.f32 %acc, {zero};
    mov.u32 %nc, 0;
SUM_LOOP:
    setp.ge.u32 %p, %nc, %NC;
    @%p bra SUM_END;

    // src_idx = nc * F + f_idx
    mad.lo.u32 %src_idx, %nc, %F, %f_idx;
    cvt.u64.u32 %addr, %src_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %mag0, %addr;
    ld.global.f32 %val, [%addr];
    add.f32 %acc, %acc, %val;

    add.u32 %nc, %nc, 1;
    bra SUM_LOOP;
SUM_END:
    mul.f32 %acc, %acc, %inv_nc;

    cvt.u64.u32 %addr, %f_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out0, %addr;
    st.global.f32 [%addr], %acc;
DONE:
    ret;
}}
"#,
        zero = f32_hex(0.0_f32)
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SM_VERSIONS: &[u32] = &[75, 80, 86, 90, 100, 120];

    fn check_kernel(name: &str, ptx: &str, sm: u32) {
        let target = format!("sm_{sm}");
        assert!(
            ptx.contains(&target),
            "kernel={name} sm={sm}: missing .target {target}"
        );
        assert!(
            ptx.contains(".visible .entry"),
            "kernel={name} sm={sm}: missing .visible .entry"
        );
        assert!(!ptx.is_empty(), "kernel={name} sm={sm}: empty PTX");
    }

    #[test]
    fn all_kernels_all_sm_versions_have_target() {
        #[allow(clippy::type_complexity)]
        let kernels: &[(&str, fn(u32) -> String)] = &[
            ("moving_average", moving_average_ptx),
            ("patch_embed_1d", patch_embed_1d_ptx),
            ("causal_temporal_conv", causal_temporal_conv_ptx),
            ("auto_correlation", auto_correlation_ptx),
            ("revin_normalize", revin_normalize_ptx),
            ("multirate_pool", multirate_pool_ptx),
            ("period_detect", period_detect_ptx),
        ];
        for &(name, kernel_fn) in kernels {
            for &sm in SM_VERSIONS {
                let ptx = kernel_fn(sm);
                check_kernel(name, &ptx, sm);
            }
        }
    }

    #[test]
    fn moving_average_ptx_contains_rcp() {
        let ptx = moving_average_ptx(80);
        assert!(ptx.contains("rcp.approx.f32"), "missing rcp instruction");
    }

    #[test]
    fn causal_temporal_conv_ptx_uses_fma() {
        let ptx = causal_temporal_conv_ptx(90);
        assert!(ptx.contains("fma.rn.f32"), "missing fma instruction");
    }

    #[test]
    fn auto_correlation_ptx_computes_magnitude() {
        let ptx = auto_correlation_ptx(80);
        assert!(ptx.contains("fma.rn.f32"), "missing fma for re*re + im*im");
    }

    #[test]
    fn revin_normalize_ptx_uses_div_approx() {
        let ptx = revin_normalize_ptx(80);
        assert!(ptx.contains("div.approx.f32"), "missing div");
    }

    #[test]
    fn multirate_pool_ptx_uses_rcp() {
        let ptx = multirate_pool_ptx(86);
        assert!(ptx.contains("rcp.approx.f32"), "missing rcp");
    }

    #[test]
    fn period_detect_ptx_has_sum_loop() {
        let ptx = period_detect_ptx(80);
        assert!(ptx.contains("SUM_LOOP"), "missing sum loop label");
    }

    #[test]
    fn patch_embed_1d_ptx_sm120() {
        let ptx = patch_embed_1d_ptx(120);
        assert!(ptx.contains("sm_120"));
        assert!(ptx.contains(".version 8.7"));
    }
}
