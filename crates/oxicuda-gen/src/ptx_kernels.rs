//! PTX GPU kernel sources for generative AI operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`ddpm_step_ptx`] | DDPM reverse diffusion step |
//! | [`cfg_combine_ptx`] | Classifier-free guidance combination |
//! | [`lora_apply_ptx`] | LoRA adapter application |
//! | [`flow_velocity_ptx`] | Flow matching Euler step |
//! | [`vae_kl_loss_ptx`] | VAE KL divergence loss |
//! | [`timestep_embed_ptx`] | Sinusoidal timestep embedding |

// ─── Hex encoding ────────────────────────────────────────────────────────────

/// Encode a `f32` as a PTX hexadecimal float literal (e.g., `0F3F800000` = 1.0f).
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

// ─── Kernel 1: ddpm_step ─────────────────────────────────────────────────────

/// DDPM reverse diffusion step kernel.
///
/// Implements: `x_prev[i] = (x_t[i] - beta/sqrt(1-alpha_bar)*eps[i]) / sqrt(alpha) + sigma*z[i]`
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_x_t` | `u64` (→ `f32*`) | Current noisy sample `x_t` |
/// | `p_eps` | `u64` (→ `f32*`) | Predicted noise `ε̂` |
/// | `p_z` | `u64` (→ `f32*`) | Standard normal noise `z` |
/// | `p_x_prev` | `u64` (→ `f32*`) | Output `x_{t-1}` |
/// | `alpha_f32` | `f32` | `α_t = 1 - β_t` |
/// | `alpha_bar_f32` | `f32` | `ᾱ_t = ∏α` |
/// | `beta_f32` | `f32` | `β_t` |
/// | `sigma_f32` | `f32` | `σ_t = sqrt(β_t*(1-ᾱ_{t-1})/(1-ᾱ_t))` |
/// | `n` | `u32` | Number of elements |
///
/// Launch: `grid = ceil(n/256)`, `block = 256`.
pub fn ddpm_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry ddpm_step(
    .param .u64 p_x_t,
    .param .u64 p_eps,
    .param .u64 p_z,
    .param .u64 p_x_prev,
    .param .f32 alpha_f32,
    .param .f32 alpha_bar_f32,
    .param .f32 beta_f32,
    .param .f32 sigma_f32,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<6>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x_t];
    ld.param.u64  %rd1, [p_eps];
    ld.param.u64  %rd2, [p_z];
    ld.param.u64  %rd3, [p_x_prev];
    ld.param.f32  %f0,  [alpha_f32];
    ld.param.f32  %f1,  [alpha_bar_f32];
    ld.param.f32  %f2,  [beta_f32];
    ld.param.f32  %f3,  [sigma_f32];
    ld.param.u32  %r0,  [n];

    // Grid-stride loop: tid = blockDim * blockIdx + threadIdx
    mov.u32        %r1, %ntid.x;
    mov.u32        %r2, %ctaid.x;
    mov.u32        %r3, %tid.x;
    mad.lo.u32     %r4, %r1, %r2, %r3;

$LOOP:
    setp.ge.u32    %p0, %r4, %r0;
    @%p0 bra $DONE;

    // Load x_t[i]
    mul.wide.u32   %rd4, %r4, 4;
    add.u64        %rd5, %rd0, %rd4;
    ld.global.f32  %f4, [%rd5];

    // Load eps[i]
    add.u64        %rd6, %rd1, %rd4;
    ld.global.f32  %f5, [%rd6];

    // Load z[i]
    add.u64        %rd7, %rd2, %rd4;
    ld.global.f32  %f6, [%rd7];

    // coeff = beta / sqrt(1 - alpha_bar)
    // 1 - alpha_bar
    mov.f32        %f7, {ONE};
    sub.f32        %f8, %f7, %f1;
    sqrt.approx.f32 %f9, %f8;
    div.approx.f32  %f10, %f2, %f9;

    // x_t - coeff * eps
    fma.rn.f32     %f11, %f10, %f5, %f4;
    sub.f32        %f11, %f4, %f10;
    // Redo: x_t - coeff * eps (manual: f11 = f4 - f10*f5)
    mul.f32        %f12, %f10, %f5;
    sub.f32        %f11, %f4, %f12;

    // divide by sqrt(alpha)
    sqrt.approx.f32 %f13, %f0;
    rcp.approx.f32  %f14, %f13;
    mul.f32        %f11, %f11, %f14;

    // + sigma * z
    fma.rn.f32     %f15, %f3, %f6, %f11;

    // Store result
    mul.wide.u32   %rd4, %r4, 4;
    add.u64        %rd5, %rd3, %rd4;
    st.global.f32  [%rd5], %f15;

    // stride = blockDim * gridDim
    mov.u32        %r5, %nctaid.x;
    mul.lo.u32     %r5, %r1, %r5;
    add.u32        %r4, %r4, %r5;
    bra $LOOP;

$DONE:
    ret;
}}
"#,
        ONE = f32_hex(1.0_f32)
    )
}

// ─── Kernel 2: cfg_combine ────────────────────────────────────────────────────

/// Classifier-free guidance combination kernel.
///
/// Implements: `out[i] = uncond[i] + scale * (cond[i] - uncond[i])`
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_cond` | `u64` (→ `f32*`) | Conditional noise prediction |
/// | `p_uncond` | `u64` (→ `f32*`) | Unconditional noise prediction |
/// | `p_out` | `u64` (→ `f32*`) | Combined output |
/// | `scale_f32` | `f32` | Guidance scale `s` |
/// | `n` | `u32` | Number of elements |
///
/// Launch: `grid = ceil(n/256)`, `block = 256`.
pub fn cfg_combine_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry cfg_combine(
    .param .u64 p_cond,
    .param .u64 p_uncond,
    .param .u64 p_out,
    .param .f32 scale_f32,
    .param .u32 n
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<5>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_cond];
    ld.param.u64  %rd1, [p_uncond];
    ld.param.u64  %rd2, [p_out];
    ld.param.f32  %f0,  [scale_f32];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

$LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $DONE;

    mul.wide.u32  %rd3, %r4, 4;

    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f2, [%rd5];

    // diff = cond - uncond
    sub.f32       %f3, %f1, %f2;
    // out = uncond + scale * diff
    fma.rn.f32    %f4, %f0, %f3, %f2;

    add.u64       %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f4;

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %nctaid.x;
    mul.lo.u32    %r2, %r1, %r2;
    add.u32       %r4, %r4, %r2;
    bra $LOOP;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 3: lora_apply ─────────────────────────────────────────────────────

/// LoRA adapter application kernel.
///
/// Implements: `out[i] = base[i] + scale * delta[i]`
/// where `scale = alpha / rank`.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_base` | `u64` (→ `f32*`) | Base weight output |
/// | `p_delta` | `u64` (→ `f32*`) | LoRA delta (B*A product) |
/// | `p_out` | `u64` (→ `f32*`) | Result buffer |
/// | `scale_f32` | `f32` | `alpha / rank` |
/// | `n` | `u32` | Number of elements |
pub fn lora_apply_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry lora_apply(
    .param .u64 p_base,
    .param .u64 p_delta,
    .param .u64 p_out,
    .param .f32 scale_f32,
    .param .u32 n
)
{{
    .reg .u64  %rd<5>;
    .reg .u32  %r<5>;
    .reg .f32  %f<5>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_base];
    ld.param.u64  %rd1, [p_delta];
    ld.param.u64  %rd2, [p_out];
    ld.param.f32  %f0,  [scale_f32];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

$LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $DONE;

    mul.wide.u32  %rd3, %r4, 4;

    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];

    // out = base + scale * delta
    fma.rn.f32    %f3, %f0, %f2, %f1;

    add.u64       %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f3;

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %nctaid.x;
    mul.lo.u32    %r2, %r1, %r2;
    add.u32       %r4, %r4, %r2;
    bra $LOOP;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 4: flow_velocity ──────────────────────────────────────────────────

/// Flow matching Euler step kernel.
///
/// Implements: `x_next[i] = x_t[i] + dt * velocity[i]`
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_x` | `u64` (→ `f32*`) | Current sample `x_t` |
/// | `p_v` | `u64` (→ `f32*`) | Velocity field |
/// | `p_out` | `u64` (→ `f32*`) | Output `x_{t+dt}` |
/// | `dt_f32` | `f32` | Step size |
/// | `n` | `u32` | Number of elements |
pub fn flow_velocity_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry flow_velocity(
    .param .u64 p_x,
    .param .u64 p_v,
    .param .u64 p_out,
    .param .f32 dt_f32,
    .param .u32 n
)
{{
    .reg .u64  %rd<5>;
    .reg .u32  %r<5>;
    .reg .f32  %f<4>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_v];
    ld.param.u64  %rd2, [p_out];
    ld.param.f32  %f0,  [dt_f32];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

$LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $DONE;

    mul.wide.u32  %rd3, %r4, 4;

    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];

    // x_next = x_t + dt * v
    fma.rn.f32    %f3, %f0, %f2, %f1;

    add.u64       %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f3;

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %nctaid.x;
    mul.lo.u32    %r2, %r1, %r2;
    add.u32       %r4, %r4, %r2;
    bra $LOOP;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 5: vae_kl_loss ────────────────────────────────────────────────────

/// VAE KL divergence loss kernel.
///
/// Implements per-element: `loss[i] = 0.5 * (mu[i]^2 + exp(logvar[i]) - 1 - logvar[i])`
///
/// Uses `ex2.approx.f32` for exp via change of base:
/// `exp(x) = exp2(x * log2(e))`.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_mu` | `u64` (→ `f32*`) | Mean μ |
/// | `p_logvar` | `u64` (→ `f32*`) | Log-variance log σ² |
/// | `p_loss` | `u64` (→ `f32*`) | Per-element KL loss |
/// | `n` | `u32` | Number of elements |
pub fn vae_kl_loss_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    // log2(e) = 1/ln(2) ≈ 1.4426950408889634
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    let half = f32_hex(0.5_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}.visible .entry vae_kl_loss(
    .param .u64 p_mu,
    .param .u64 p_logvar,
    .param .u64 p_loss,
    .param .u32 n
)
{{
    .reg .u64  %rd<5>;
    .reg .u32  %r<5>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_mu];
    ld.param.u64  %rd1, [p_logvar];
    ld.param.u64  %rd2, [p_loss];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

$LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $DONE;

    mul.wide.u32  %rd3, %r4, 4;

    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];

    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    // mu^2
    mul.f32       %f2, %f0, %f0;

    // exp(logvar): exp(x) = exp2(x * log2(e))
    mul.f32       %f3, %f1, {LOG2E};
    ex2.approx.f32 %f4, %f3;

    // 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    add.f32       %f5, %f2, %f4;
    sub.f32       %f5, %f5, {ONE};
    sub.f32       %f5, %f5, %f1;
    mul.f32       %f6, {HALF}, %f5;

    add.u64       %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f6;

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %nctaid.x;
    mul.lo.u32    %r2, %r1, %r2;
    add.u32       %r4, %r4, %r2;
    bra $LOOP;

$DONE:
    ret;
}}
"#,
        LOG2E = log2e,
        HALF = half,
        ONE = one,
    )
}

// ─── Kernel 6: timestep_embed ─────────────────────────────────────────────────

/// Sinusoidal timestep embedding kernel.
///
/// For each `(t_idx, i)`:
/// - `i < half_dim`: `emb[t_idx*dim + i] = sin(t / max_period^(2i/dim))`
/// - `i >= half_dim`: `emb[t_idx*dim + i] = cos(t / max_period^(2(i-half_dim)/dim))`
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_timesteps` | `u64` (→ `f32*`) | Array of timestep values |
/// | `p_out` | `u64` (→ `f32*`) | Output embeddings |
/// | `dim` | `u32` | Total embedding dimension (must be even) |
/// | `n_timesteps` | `u32` | Number of timesteps |
/// | `max_period_f32` | `f32` | Max period (typically 10000.0) |
pub fn timestep_embed_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let two = f32_hex(2.0_f32);
    format!(
        r#"{hdr}.visible .entry timestep_embed(
    .param .u64 p_timesteps,
    .param .u64 p_out,
    .param .u32 dim,
    .param .u32 n_timesteps,
    .param .f32 max_period_f32
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<14>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_timesteps];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [dim];
    ld.param.u32  %r1,  [n_timesteps];
    ld.param.f32  %f0,  [max_period_f32];

    // half_dim = dim / 2
    shr.u32       %r8, %r0, 1;

    // total = n_timesteps * dim
    mul.lo.u32    %r9, %r1, %r0;

    // tid = blockDim * blockIdx + threadIdx
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

$LOOP:
    setp.ge.u32   %p0, %r5, %r9;
    @%p0 bra $DONE;

    // t_idx = tid / dim
    div.u32       %r6, %r5, %r0;
    // dim_idx = tid % dim
    rem.u32       %r7, %r5, %r0;

    // Load timestep t
    mul.wide.u32  %rd2, %r6, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];

    // Determine if sin or cos: sin for i < half_dim, cos otherwise
    // freq_idx = dim_idx if dim_idx < half_dim, else dim_idx - half_dim
    setp.lt.u32   %p1, %r7, %r8;
    @%p1 bra $SIN_BRANCH;

    // cos branch: freq_idx = dim_idx - half_dim
    sub.u32       %r7, %r7, %r8;

$SIN_BRANCH:
    // freq = 2 * freq_idx / dim (as f32)
    cvt.rn.f32.u32 %f2, %r7;
    cvt.rn.f32.u32 %f3, %r0;
    mul.f32        %f4, {TWO}, %f2;
    div.approx.f32 %f5, %f4, %f3;

    // exponent = -freq * log2(max_period) using lg2
    // max_period^(freq) = exp(freq * ln(max_period))
    // = exp2(freq * log2(max_period))
    // We want 1 / max_period^freq = exp2(-freq * log2(max_period))
    lg2.approx.f32 %f6, %f0;
    mul.f32        %f7, %f5, %f6;
    neg.f32        %f7, %f7;
    ex2.approx.f32 %f8, %f7;

    // angle = t * inv_freq
    mul.f32        %f9, %f1, %f8;

    // apply sin or cos
    @%p1 bra $DO_SIN;
    cos.approx.f32 %f10, %f9;
    bra $STORE;

$DO_SIN:
    sin.approx.f32 %f10, %f9;

$STORE:
    // out_idx = t_idx * dim + original_dim_idx (recompute)
    rem.u32        %r7, %r5, %r0;
    mad.lo.u32     %r6, %r6, %r0, %r7;
    mul.wide.u32   %rd4, %r6, 4;
    add.u64        %rd5, %rd1, %rd4;
    st.global.f32  [%rd5], %f10;

    // stride
    mov.u32        %r2, %ntid.x;
    mov.u32        %r3, %nctaid.x;
    mul.lo.u32     %r3, %r2, %r3;
    add.u32        %r5, %r5, %r3;
    bra $LOOP;

$DONE:
    ret;
}}
"#,
        TWO = two,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SM_VERSIONS: &[u32] = &[75, 80, 86, 90, 100, 120];

    fn check_kernel(ptx: &str, sm: u32, entry_name: &str) {
        assert!(
            ptx.contains(&format!(".target sm_{sm}")),
            "missing .target sm_{sm}"
        );
        assert!(ptx.contains(entry_name), "missing entry: {entry_name}");
        assert!(ptx.contains(".address_size 64"), "missing .address_size 64");
    }

    #[test]
    fn f32_hex_one() {
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }

    #[test]
    fn f32_hex_zero() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
    }

    #[test]
    fn f32_hex_neg_one() {
        assert_eq!(f32_hex(-1.0_f32), "0FBF800000");
    }

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        for &sm in SM_VERSIONS {
            let ptx = ddpm_step_ptx(sm);
            check_kernel(&ptx, sm, "ddpm_step");

            let ptx = cfg_combine_ptx(sm);
            check_kernel(&ptx, sm, "cfg_combine");

            let ptx = lora_apply_ptx(sm);
            check_kernel(&ptx, sm, "lora_apply");

            let ptx = flow_velocity_ptx(sm);
            check_kernel(&ptx, sm, "flow_velocity");

            let ptx = vae_kl_loss_ptx(sm);
            check_kernel(&ptx, sm, "vae_kl_loss");

            let ptx = timestep_embed_ptx(sm);
            check_kernel(&ptx, sm, "timestep_embed");
        }
    }

    #[test]
    fn ddpm_step_ptx_contains_sqrt_rcp() {
        let ptx = ddpm_step_ptx(80);
        assert!(ptx.contains("sqrt.approx.f32"), "missing sqrt");
        assert!(ptx.contains("rcp.approx.f32"), "missing rcp");
        assert!(ptx.contains("fma.rn.f32"), "missing fma");
    }

    #[test]
    fn cfg_combine_ptx_contains_fma() {
        let ptx = cfg_combine_ptx(80);
        assert!(ptx.contains("fma.rn.f32"), "missing fma");
        assert!(ptx.contains("sub.f32"), "missing sub");
    }

    #[test]
    fn lora_apply_ptx_structure() {
        let ptx = lora_apply_ptx(90);
        assert!(ptx.contains("lora_apply"), "missing entry");
        assert!(ptx.contains("scale_f32"), "missing scale_f32 param");
    }

    #[test]
    fn flow_velocity_ptx_structure() {
        let ptx = flow_velocity_ptx(86);
        assert!(ptx.contains("flow_velocity"), "missing entry");
        assert!(ptx.contains("dt_f32"), "missing dt_f32 param");
    }

    #[test]
    fn vae_kl_loss_uses_ex2() {
        let ptx = vae_kl_loss_ptx(80);
        assert!(ptx.contains("ex2.approx.f32"), "missing ex2 for exp");
    }

    #[test]
    fn timestep_embed_uses_sin_cos() {
        let ptx = timestep_embed_ptx(80);
        assert!(ptx.contains("sin.approx.f32"), "missing sin");
        assert!(ptx.contains("cos.approx.f32"), "missing cos");
        assert!(ptx.contains("lg2.approx.f32"), "missing lg2");
    }

    #[test]
    fn ptx_version_per_sm() {
        assert!(ddpm_step_ptx(75).contains(".version 7.5"));
        assert!(ddpm_step_ptx(80).contains(".version 8.0"));
        assert!(ddpm_step_ptx(90).contains(".version 8.4"));
        assert!(ddpm_step_ptx(100).contains(".version 8.7"));
    }
}
