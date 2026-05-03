//! PTX GPU kernel sources for Bayesian deep learning operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`kl_gaussian_ptx`] | Per-element KL(N(μ,σ²) ‖ N(0,1)) with atomic accumulation |
//! | [`mc_dropout_mask_ptx`] | Bernoulli dropout mask via inline LCG |
//! | [`local_reparam_ptx`] | Local reparameterization with Box-Muller sampling |
//! | [`ece_bucket_ptx`] | ECE histogram binning with atomic counters |
//! | [`ensemble_aggregate_ptx`] | Ensemble mean/variance over M member logits |
//! | [`flipout_perturb_ptx`] | Flipout ±1 sign perturbation for efficient Bayesian inference |
//! | [`temp_scale_logits_ptx`] | Temperature scaling of logits |

// ─── PTX header helper ────────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

/// Format an f32 as a PTX hex literal.
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: kl_gaussian ───────────────────────────────────────────────────

/// Per-element KL divergence KL(N(μ,σ²) ‖ N(0,1)).
///
/// Computes `0.5 * (μ² + σ² - 1 - ln(σ²))` where `σ = exp(log_sigma)`,
/// using `ex2.approx.f32` and `lg2.approx.f32`. Partial sums are accumulated
/// into a scalar output via `atom.global.add.f32`.
#[must_use]
pub fn kl_gaussian_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let half = f32_hex(0.5_f32);
    let one = f32_hex(1.0_f32);
    let log2e = f32_hex(core::f32::consts::LOG2_E);
    format!(
        r#"{hdr}// kl_gaussian_kernel: KL(N(mu, sigma^2) || N(0,1))
// For each element i: contrib = 0.5*(mu[i]^2 + sigma[i]^2 - 1 - ln(sigma[i]^2))
// sigma[i] = exp(log_sigma[i])
// ln(sigma^2) = 2 * log_sigma  = 2 * log_sigma[i]
// contrib = 0.5*(mu^2 + exp(2*log_sigma) - 1 - 2*log_sigma)
// Uses ex2.approx (base 2) and lg2.approx (base 2); multiply by ln(2) for natural.
.visible .entry kl_gaussian_kernel(
    .param .u64 p_mu,
    .param .u64 p_log_sigma,
    .param .u64 p_out_kl,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<12>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_mu];
    ld.param.u64  %rd1, [p_log_sigma];
    ld.param.u64  %rd2, [p_out_kl];
    ld.param.u32  %r0,  [n];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r1,  %ntid.x;
    mov.u32       %r2,  %ctaid.x;
    mov.u32       %r3,  %tid.x;
    mad.lo.u32    %r4,  %r1, %r2, %r3;    // r4 = tid

    // grid-stride loop
    mul.lo.u32    %r5,  %r1, %r2;         // used for step: gridDim*blockDim
    // step = ntid.x * nctaid.x
    mov.u32       %r6,  %nctaid.x;
    mul.lo.u32    %r5,  %r1, %r6;         // r5 = grid_stride

    mov.u32       %r7,  %r4;              // i = tid

$KLG_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $KLG_DONE;

    // Load mu[i] and log_sigma[i]
    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];            // f0 = mu

    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f1, [%rd5];            // f1 = log_sigma

    // mu^2
    mul.f32       %f2, %f0, %f0;          // f2 = mu^2

    // exp(2 * log_sigma) via ex2.approx:
    // exp(x) = ex2(x * log2(e))
    // x = 2 * log_sigma
    add.f32       %f3, %f1, %f1;          // f3 = 2 * log_sigma
    mul.f32       %f4, %f3, {LOG2E};      // f4 = 2*log_sigma * log2(e)
    ex2.approx.f32 %f5, %f4;             // f5 = exp(2*log_sigma) = sigma^2

    // ln(sigma^2) = 2*log_sigma (already computed as f3 in natural log since log_sigma is natural)
    // but we need natural log from f3 which is already natural: f3 = 2*log_sigma (natural)
    // so ln_sigma_sq = f3

    // contrib = 0.5*(mu^2 + sigma^2 - 1 - ln(sigma^2))
    add.f32       %f6, %f2, %f5;         // mu^2 + sigma^2
    sub.f32       %f6, %f6, {ONE};       // - 1
    sub.f32       %f6, %f6, %f3;         // - 2*log_sigma (= ln(sigma^2))
    mul.f32       %f6, %f6, {HALF};      // * 0.5

    // Atomic add to output scalar
    atom.global.add.f32 %f7, [%rd2], %f6;

    add.u32       %r7, %r7, %r5;
    bra           $KLG_LOOP;

$KLG_DONE:
    ret;
}}
"#,
        HALF = half,
        ONE = one,
        LOG2E = log2e,
    )
}

// ─── Kernel 2: mc_dropout_mask ────────────────────────────────────────────────

/// Bernoulli dropout mask using inline LCG: `mask[i] = (lcg(seed,i) > drop_rate) ? 1/keep : 0`.
#[must_use]
pub fn mc_dropout_mask_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// mc_dropout_mask_kernel: Bernoulli mask using LCG per-element.
// mask[i] = (lcg_rand(seed, i) > drop_rate) ? 1.0/keep_rate : 0.0
// keep_rate = 1.0 - drop_rate
// LCG: state = A*state + C (mod 2^64), A=6364136223846793005, C=1442695040888963407
// seed is per-call; element index XORed in for diversity.
.visible .entry mc_dropout_mask_kernel(
    .param .u64 p_mask,
    .param .u32 n,
    .param .f32 drop_rate,
    .param .u64 seed
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_mask];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [drop_rate];     // f0 = drop_rate
    ld.param.u64  %rd1, [seed];          // rd1 = seed

    // keep_rate = 1.0 - drop_rate; scale = 1.0 / keep_rate
    mov.f32       %f1, {ONE};
    sub.f32       %f2, %f1, %f0;         // f2 = keep_rate
    div.rn.f32    %f3, %f1, %f2;         // f3 = 1/keep_rate

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r1,  %ntid.x;
    mov.u32       %r2,  %ctaid.x;
    mov.u32       %r3,  %tid.x;
    mad.lo.u32    %r4,  %r1, %r2, %r3;

    mov.u32       %r6,  %nctaid.x;
    mul.lo.u32    %r5,  %r1, %r6;        // r5 = grid_stride

    mov.u32       %r7,  %r4;             // i = tid

$MCD_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $MCD_DONE;

    // LCG: state = seed ^ i, then advance
    cvt.u64.u32   %rd2, %r7;             // rd2 = (u64)i
    xor.b64       %rd3, %rd1, %rd2;      // rd3 = seed ^ i
    // LCG step: rd3 = A * rd3 + C
    mov.u64       %rd4, 6364136223846793005;
    mul.lo.u64    %rd3, %rd3, %rd4;
    mov.u64       %rd5, 1442695040888963407;
    add.u64       %rd3, %rd3, %rd5;
    // Extract high 32 bits as random u32
    shr.u64       %rd6, %rd3, 33;
    cvt.u32.u64   %r8,  %rd6;

    // Convert to f32 in [0,1): r8 / (2^31)
    cvt.rn.f32.u32 %f4, %r8;
    mov.f32        %f5, 0F4F000000;      // 2^31 as float
    div.rn.f32     %f4, %f4, %f5;       // f4 = uniform [0,1)

    // mask = (f4 > drop_rate) ? 1/keep : 0
    setp.gt.f32   %p1, %f4, %f0;
    selp.f32      %f6, %f3, {ZERO}, %p1;

    mul.wide.u32  %rd7, %r7, 4;
    add.u64       %rd7, %rd0, %rd7;
    st.global.f32 [%rd7], %f6;

    add.u32       %r7, %r7, %r5;
    bra           $MCD_LOOP;

$MCD_DONE:
    ret;
}}
"#,
        ONE = one,
        ZERO = zero,
    )
}

// ─── Kernel 3: local_reparam ──────────────────────────────────────────────────

/// Local reparameterization: given `W_mu[i]`, `W_log_var[i]`, `x[i]`,
/// compute `act_mu[i] = W_mu[i]*x[i]`, `act_var[i] = exp(W_log_var[i])*x[i]²`,
/// then sample `z[i] = act_mu + sqrt(act_var) * eps` where eps is via Box-Muller.
#[must_use]
pub fn local_reparam_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let log2e = f32_hex(core::f32::consts::LOG2_E);
    let two_pi = f32_hex(2.0 * std::f32::consts::PI);
    let eps_floor = f32_hex(1e-6_f32);
    let one = f32_hex(1.0_f32);
    let neg2 = f32_hex(-2.0_f32);
    format!(
        r#"{hdr}// local_reparam_kernel: local reparameterization trick.
// act_mu[i] = W_mu[i] * x[i]
// act_var[i] = exp(W_log_var[i]) * x[i]^2
// z[i] = act_mu[i] + sqrt(act_var[i]) * eps   (Box-Muller, two seeds)
.visible .entry local_reparam_kernel(
    .param .u64 p_w_mu,
    .param .u64 p_w_log_var,
    .param .u64 p_x,
    .param .u64 p_z,
    .param .u32 n,
    .param .u32 seed1,
    .param .u32 seed2
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<20>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_w_mu];
    ld.param.u64  %rd1, [p_w_log_var];
    ld.param.u64  %rd2, [p_x];
    ld.param.u64  %rd3, [p_z];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [seed1];
    ld.param.u32  %r2,  [seed2];

    mov.u32       %r3,  %ntid.x;
    mov.u32       %r4,  %ctaid.x;
    mov.u32       %r5,  %tid.x;
    mad.lo.u32    %r6,  %r3, %r4, %r5;

    mov.u32       %r8,  %nctaid.x;
    mul.lo.u32    %r7,  %r3, %r8;

    mov.u32       %r9,  %r6;

$LRP_LOOP:
    setp.ge.u32   %p0, %r9, %r0;
    @%p0 bra $LRP_DONE;

    mul.wide.u32  %rd4, %r9, 4;

    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f0,  [%rd5];          // f0 = W_mu[i]

    add.u64       %rd6, %rd1, %rd4;
    ld.global.f32 %f1,  [%rd6];          // f1 = W_log_var[i]

    add.u64       %rd7, %rd2, %rd4;
    ld.global.f32 %f2,  [%rd7];          // f2 = x[i]

    // act_mu = W_mu * x
    mul.f32       %f3, %f0, %f2;         // f3 = act_mu

    // act_var = exp(W_log_var) * x^2
    mul.f32       %f4, %f1, {LOG2E};
    ex2.approx.f32 %f5, %f4;            // f5 = exp(W_log_var)
    mul.f32       %f6, %f2, %f2;         // x^2
    mul.f32       %f7, %f5, %f6;         // act_var

    // Box-Muller for eps: u1, u2 from LCG(seed1^i, seed2^i)
    cvt.u64.u32   %rd8, %r9;
    // u1 from seed1 ^ i
    mov.u64       %rd9, 6364136223846793005;
    cvt.u64.u32   %rd10, %r1;
    xor.b64       %rd10, %rd10, %rd8;
    mul.lo.u64    %rd10, %rd10, %rd9;
    mov.u64       %rd11, 1442695040888963407;
    add.u64       %rd10, %rd10, %rd11;
    shr.u64       %rd10, %rd10, 33;
    cvt.u32.u64   %r10, %rd10;
    cvt.rn.f32.u32 %f8, %r10;
    mov.f32        %f9, 0F4F000000;
    div.rn.f32     %f8, %f8, %f9;
    // clamp u1 to (eps_floor, 1-eps_floor)
    max.f32        %f8, %f8, {EPS_FLOOR};
    sub.f32        %f10, {ONE}, {EPS_FLOOR};
    min.f32        %f8, %f8, %f10;

    // u2 from seed2 ^ i
    cvt.u64.u32   %rd10, %r2;
    xor.b64       %rd10, %rd10, %rd8;
    mul.lo.u64    %rd10, %rd10, %rd9;
    add.u64       %rd10, %rd10, %rd11;
    shr.u64       %rd10, %rd10, 33;
    cvt.u32.u64   %r11, %rd10;
    cvt.rn.f32.u32 %f11, %r11;
    div.rn.f32     %f11, %f11, %f9;

    // eps = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    // ln(u1) = lg2(u1) * ln(2)
    lg2.approx.f32 %f12, %f8;
    mul.f32       %f12, %f12, {LN2_CONST};   // ln(u1)
    mul.f32       %f12, %f12, {NEG2};         // -2*ln(u1)
    sqrt.approx.f32 %f12, %f12;              // sqrt(-2*ln(u1))

    mul.f32       %f13, %f11, {TWO_PI};
    cos.approx.f32 %f13, %f13;              // cos(2*pi*u2)
    mul.f32       %f13, %f12, %f13;          // eps = sqrt(-2ln(u1))*cos(2pi*u2)

    // z = act_mu + sqrt(act_var) * eps
    sqrt.approx.f32 %f14, %f7;              // sqrt(act_var)
    fma.rn.f32    %f15, %f14, %f13, %f3;    // z = act_mu + sqrt(act_var)*eps

    add.u64       %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f15;

    add.u32       %r9, %r9, %r7;
    bra           $LRP_LOOP;

$LRP_DONE:
    ret;
}}
"#,
        LOG2E = log2e,
        TWO_PI = two_pi,
        EPS_FLOOR = eps_floor,
        ONE = one,
        NEG2 = neg2,
        LN2_CONST = f32_hex(core::f32::consts::LN_2),
    )
}

// ─── Kernel 4: ece_bucket ────────────────────────────────────────────────────

/// ECE histogram binning: for each (confidence, correct) sample, atomically
/// increments `count[bin]`, `sum_conf[bin]`, and `sum_correct[bin]`.
#[must_use]
pub fn ece_bucket_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let one_f = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// ece_bucket_kernel: histogram binning for ECE computation.
// bin = min(floor(confidence[i] * n_bins), n_bins-1)
// atom.global.add.u32 count[bin], 1
// atom.global.add.f32 sum_conf[bin], confidence[i]
// atom.global.add.f32 sum_correct[bin], (correct[i] ? 1.0 : 0.0)
.visible .entry ece_bucket_kernel(
    .param .u64 p_confidence,
    .param .u64 p_correct,
    .param .u64 p_count,
    .param .u64 p_sum_conf,
    .param .u64 p_sum_correct,
    .param .u32 n,
    .param .u32 n_bins
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<20>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_confidence];
    ld.param.u64  %rd1, [p_correct];
    ld.param.u64  %rd2, [p_count];
    ld.param.u64  %rd3, [p_sum_conf];
    ld.param.u64  %rd4, [p_sum_correct];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [n_bins];

    mov.u32       %r2,  %ntid.x;
    mov.u32       %r3,  %ctaid.x;
    mov.u32       %r4,  %tid.x;
    mad.lo.u32    %r5,  %r2, %r3, %r4;

    mov.u32       %r7,  %nctaid.x;
    mul.lo.u32    %r6,  %r2, %r7;

    mov.u32       %r8,  %r5;

$ECE_LOOP:
    setp.ge.u32   %p0, %r8, %r0;
    @%p0 bra $ECE_DONE;

    mul.wide.u32  %rd5, %r8, 4;

    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f0, [%rd6];           // f0 = confidence[i]

    add.u64       %rd7, %rd1, %rd5;
    ld.global.u32 %r9, [%rd7];           // r9 = correct[i] (0 or 1)

    // bin = floor(confidence * n_bins), clamped to [0, n_bins-1]
    cvt.rn.f32.u32 %f1, %r1;            // f1 = (f32)n_bins
    mul.f32        %f2, %f0, %f1;
    cvt.rzi.u32.f32 %r10, %f2;          // floor
    sub.u32        %r11, %r1, 1;
    min.u32        %r10, %r10, %r11;    // clamp

    // byte offset = bin * 4
    mul.wide.u32  %rd8, %r10, 4;

    // count[bin]++
    add.u64       %rd9, %rd2, %rd8;
    atom.global.add.u32 %r12, [%rd9], 1;

    // sum_conf[bin] += confidence
    add.u64       %rd10, %rd3, %rd8;
    atom.global.add.f32 %f3, [%rd10], %f0;

    // sum_correct[bin] += (correct ? 1.0 : 0.0)
    setp.ne.u32   %p1, %r9, 0;
    selp.f32      %f4, {ONE_F}, 0F00000000, %p1;
    add.u64       %rd11, %rd4, %rd8;
    atom.global.add.f32 %f5, [%rd11], %f4;

    add.u32       %r8, %r8, %r6;
    bra           $ECE_LOOP;

$ECE_DONE:
    ret;
}}
"#,
        ONE_F = one_f,
    )
}

// ─── Kernel 5: ensemble_aggregate ────────────────────────────────────────────

/// Ensemble mean/variance over M member logits `[M × C]`;
/// computes `mean[c]` and Bessel-corrected `var[c]` over the M members.
#[must_use]
pub fn ensemble_aggregate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// ensemble_aggregate_kernel: mean and Bessel-corrected variance.
// Input: logits[M * C] row-major (member m, class c → offset m*C + c)
// Output: mean[C], var[C]
// One thread per class c.
.visible .entry ensemble_aggregate_kernel(
    .param .u64 p_logits,
    .param .u64 p_mean,
    .param .u64 p_var,
    .param .u32 M,
    .param .u32 C
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<16>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_logits];
    ld.param.u64  %rd1, [p_mean];
    ld.param.u64  %rd2, [p_var];
    ld.param.u32  %r0,  [M];
    ld.param.u32  %r1,  [C];

    mov.u32       %r2,  %ntid.x;
    mov.u32       %r3,  %ctaid.x;
    mov.u32       %r4,  %tid.x;
    mad.lo.u32    %r5,  %r2, %r3, %r4;   // tid = c

    setp.ge.u32   %p0, %r5, %r1;
    @%p0 bra $ENS_DONE;

    // Pass 1: compute mean over M
    mov.f32       %f0, {ZERO};            // sum = 0
    mov.u32       %r6, 0;                 // m = 0

$ENS_MEAN_LOOP:
    setp.ge.u32   %p1, %r6, %r0;
    @%p1 bra $ENS_MEAN_END;

    // offset = m * C + c
    mad.lo.u32    %r7, %r6, %r1, %r5;
    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32       %f0, %f0, %f1;

    add.u32       %r6, %r6, 1;
    bra           $ENS_MEAN_LOOP;

$ENS_MEAN_END:
    cvt.rn.f32.u32 %f2, %r0;             // f2 = (f32)M
    div.rn.f32    %f3, %f0, %f2;         // f3 = mean

    mul.wide.u32  %rd5, %r5, 4;
    add.u64       %rd6, %rd1, %rd5;
    st.global.f32 [%rd6], %f3;

    // Pass 2: Bessel-corrected variance: sum (logit - mean)^2 / (M-1)
    mov.f32       %f4, {ZERO};
    mov.u32       %r6, 0;

$ENS_VAR_LOOP:
    setp.ge.u32   %p1, %r6, %r0;
    @%p1 bra $ENS_VAR_END;

    mad.lo.u32    %r7, %r6, %r1, %r5;
    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f5, [%rd4];
    sub.f32       %f6, %f5, %f3;
    fma.rn.f32    %f4, %f6, %f6, %f4;   // sum += (x-mean)^2

    add.u32       %r6, %r6, 1;
    bra           $ENS_VAR_LOOP;

$ENS_VAR_END:
    // var = sum / (M - 1) if M > 1, else 0
    sub.u32       %r8, %r0, 1;           // M - 1
    setp.eq.u32   %p1, %r8, 0;
    @%p1 bra $ENS_ZERO_VAR;

    cvt.rn.f32.u32 %f7, %r8;
    div.rn.f32    %f8, %f4, %f7;
    bra $ENS_STORE_VAR;

$ENS_ZERO_VAR:
    mov.f32       %f8, {ZERO};

$ENS_STORE_VAR:
    add.u64       %rd7, %rd2, %rd5;
    st.global.f32 [%rd7], %f8;

$ENS_DONE:
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 6: flipout_perturb ────────────────────────────────────────────────

/// Flipout perturbation: sample random signs `r[i] ∈ {-1,+1}` and `s[j] ∈ {-1,+1}` via LCG,
/// compute `delta_out[j] += s[j] * Σ_i (W_delta[j,i] * r[i] * x[i])`.
#[must_use]
pub fn flipout_perturb_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let pos1 = f32_hex(1.0_f32);
    let neg1 = f32_hex(-1.0_f32);
    format!(
        r#"{hdr}// flipout_perturb_kernel: Flipout ±1 sign perturbation.
// For each output j:
//   s_j = sign from LCG bit
//   delta_out[j] = s_j * sum_i(W_delta[j,i] * r_i * x[i])
// where r_i = sign from LCG bit
// W_delta is [out_features × in_features] row-major.
.visible .entry flipout_perturb_kernel(
    .param .u64 p_w_delta,
    .param .u64 p_x,
    .param .u64 p_r_signs,
    .param .u64 p_delta_out,
    .param .u32 out_features,
    .param .u32 in_features,
    .param .u64 seed_s
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<20>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_w_delta];
    ld.param.u64  %rd1, [p_x];
    ld.param.u64  %rd2, [p_r_signs];
    ld.param.u64  %rd3, [p_delta_out];
    ld.param.u32  %r0,  [out_features];
    ld.param.u32  %r1,  [in_features];
    ld.param.u64  %rd4, [seed_s];

    // tid = j (output index)
    mov.u32       %r2,  %ntid.x;
    mov.u32       %r3,  %ctaid.x;
    mov.u32       %r4,  %tid.x;
    mad.lo.u32    %r5,  %r2, %r3, %r4;

    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $FPO_DONE;

    // Compute s_j from seed_s ^ j using LCG bit
    cvt.u64.u32   %rd5, %r5;
    xor.b64       %rd5, %rd4, %rd5;
    mov.u64       %rd6, 6364136223846793005;
    mul.lo.u64    %rd5, %rd5, %rd6;
    mov.u64       %rd7, 1442695040888963407;
    add.u64       %rd5, %rd5, %rd7;
    shr.u64       %rd5, %rd5, 33;
    cvt.u32.u64   %r6,  %rd5;
    // s_j = bit 0 of r6: 1 → +1.0, 0 → -1.0
    and.b32       %r7, %r6, 1;
    setp.eq.u32   %p1, %r7, 1;
    selp.f32      %f0, {POS1}, {NEG1}, %p1;   // f0 = s_j

    // Inner loop over in_features
    mov.f32       %f1, 0F00000000;             // acc = 0
    mov.u32       %r8, 0;                       // i = 0

$FPO_INNER:
    setp.ge.u32   %p1, %r8, %r1;
    @%p1 bra $FPO_INNER_END;

    // W_delta[j * in_features + i]
    mad.lo.u32    %r9, %r5, %r1, %r8;
    mul.wide.u32  %rd8, %r9, 4;
    add.u64       %rd9, %rd0, %rd8;
    ld.global.f32 %f2, [%rd9];

    // x[i]
    mul.wide.u32  %rd10, %r8, 4;
    add.u64       %rd11, %rd1, %rd10;
    ld.global.f32 %f3, [%rd11];

    // r_signs[i]
    add.u64       %rd12, %rd2, %rd10;
    ld.global.f32 %f4, [%rd12];

    // acc += W_delta[j,i] * r_i * x[i]
    mul.f32       %f5, %f2, %f4;
    fma.rn.f32    %f1, %f5, %f3, %f1;

    add.u32       %r8, %r8, 1;
    bra           $FPO_INNER;

$FPO_INNER_END:
    // delta_out[j] = s_j * acc
    mul.f32       %f6, %f0, %f1;

    mul.wide.u32  %rd13, %r5, 4;
    add.u64       %rd14, %rd3, %rd13;
    st.global.f32 [%rd14], %f6;

$FPO_DONE:
    ret;
}}
"#,
        POS1 = pos1,
        NEG1 = neg1,
    )
}

// ─── Kernel 7: temp_scale_logits ─────────────────────────────────────────────

/// Temperature scaling: `out[i] = logit[i] / temperature`, grid-stride loop.
#[must_use]
pub fn temp_scale_logits_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// temp_scale_logits_kernel: out[i] = logit[i] / temperature
.visible .entry temp_scale_logits_kernel(
    .param .u64 p_logits,
    .param .u64 p_out,
    .param .u32 n,
    .param .f32 temperature
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<12>;
    .reg .f32  %f<4>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_logits];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [temperature];   // f0 = T

    mov.u32       %r1,  %ntid.x;
    mov.u32       %r2,  %ctaid.x;
    mov.u32       %r3,  %tid.x;
    mad.lo.u32    %r4,  %r1, %r2, %r3;

    mov.u32       %r6,  %nctaid.x;
    mul.lo.u32    %r5,  %r1, %r6;

    mov.u32       %r7,  %r4;

$TSL_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $TSL_DONE;

    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];

    div.rn.f32    %f2, %f1, %f0;         // logit / T

    add.u64       %rd4, %rd1, %rd2;
    st.global.f32 [%rd4], %f2;

    add.u32       %r7, %r7, %r5;
    bra           $TSL_LOOP;

$TSL_DONE:
    ret;
}}
"#
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_SM: &[u32] = &[75, 80, 86, 90, 100, 120];

    fn check_kernel(ptx: &str, sm: u32) {
        assert!(
            ptx.contains(&format!(".target sm_{sm}")),
            "missing .target sm_{sm}"
        );
        assert!(ptx.contains(".address_size 64"), "missing .address_size 64");
    }

    #[test]
    fn kl_gaussian_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&kl_gaussian_ptx(sm), sm);
        }
    }

    #[test]
    fn mc_dropout_mask_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&mc_dropout_mask_ptx(sm), sm);
        }
    }

    #[test]
    fn local_reparam_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&local_reparam_ptx(sm), sm);
        }
    }

    #[test]
    fn ece_bucket_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&ece_bucket_ptx(sm), sm);
        }
    }

    #[test]
    fn ensemble_aggregate_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&ensemble_aggregate_ptx(sm), sm);
        }
    }

    #[test]
    fn flipout_perturb_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&flipout_perturb_ptx(sm), sm);
        }
    }

    #[test]
    fn temp_scale_logits_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&temp_scale_logits_ptx(sm), sm);
        }
    }

    #[test]
    fn kl_gaussian_has_atomic_add() {
        assert!(kl_gaussian_ptx(80).contains("atom.global.add.f32"));
    }

    #[test]
    fn kl_gaussian_has_ex2_lg2() {
        let p = kl_gaussian_ptx(80);
        assert!(p.contains("ex2.approx.f32"));
    }

    #[test]
    fn mc_dropout_has_selp() {
        assert!(mc_dropout_mask_ptx(80).contains("selp.f32"));
    }

    #[test]
    fn local_reparam_has_box_muller() {
        let p = local_reparam_ptx(80);
        assert!(p.contains("sqrt.approx.f32") && p.contains("cos.approx.f32"));
    }

    #[test]
    fn ece_bucket_has_atomic_add_u32() {
        assert!(ece_bucket_ptx(80).contains("atom.global.add.u32"));
    }

    #[test]
    fn ensemble_aggregate_has_fma() {
        assert!(ensemble_aggregate_ptx(80).contains("fma.rn.f32"));
    }

    #[test]
    fn flipout_perturb_has_entry_name() {
        assert!(flipout_perturb_ptx(80).contains(".visible .entry flipout_perturb_kernel"));
    }

    #[test]
    fn temp_scale_has_div() {
        assert!(temp_scale_logits_ptx(80).contains("div.rn.f32"));
    }

    #[test]
    fn sm120_uses_ptx_87() {
        for ptx in [
            kl_gaussian_ptx(120),
            mc_dropout_mask_ptx(120),
            local_reparam_ptx(120),
            ece_bucket_ptx(120),
            ensemble_aggregate_ptx(120),
            flipout_perturb_ptx(120),
            temp_scale_logits_ptx(120),
        ] {
            assert!(ptx.contains(".version 8.7"), "sm_120 must use PTX 8.7");
        }
    }

    #[test]
    fn sm90_uses_ptx_84() {
        for ptx in [
            kl_gaussian_ptx(90),
            mc_dropout_mask_ptx(90),
            local_reparam_ptx(90),
            ece_bucket_ptx(90),
            ensemble_aggregate_ptx(90),
            flipout_perturb_ptx(90),
            temp_scale_logits_ptx(90),
        ] {
            assert!(ptx.contains(".version 8.4"), "sm_90 must use PTX 8.4");
        }
    }

    #[test]
    fn sm80_uses_ptx_80() {
        for ptx in [
            kl_gaussian_ptx(80),
            mc_dropout_mask_ptx(80),
            local_reparam_ptx(80),
            ece_bucket_ptx(80),
            ensemble_aggregate_ptx(80),
            flipout_perturb_ptx(80),
            temp_scale_logits_ptx(80),
        ] {
            assert!(ptx.contains(".version 8.0"), "sm_80 must use PTX 8.0");
        }
    }

    #[test]
    fn sm75_uses_ptx_75() {
        for ptx in [
            kl_gaussian_ptx(75),
            mc_dropout_mask_ptx(75),
            local_reparam_ptx(75),
            ece_bucket_ptx(75),
            ensemble_aggregate_ptx(75),
            flipout_perturb_ptx(75),
            temp_scale_logits_ptx(75),
        ] {
            assert!(ptx.contains(".version 7.5"), "sm_75 must use PTX 7.5");
        }
    }
}
