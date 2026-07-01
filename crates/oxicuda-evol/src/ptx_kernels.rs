//! GPU PTX kernels for Evolutionary Algorithm operations.
//!
//! Each function returns a complete PTX module string parameterised on SM version.
//! PTX ISA is selected by SM:
//!   SM>=100 -> 8.7 (Blackwell), SM>=90 -> 8.4 (Hopper),
//!   SM>=80  -> 8.0 (Ampere),   else    -> 7.5 (Turing).
//!
//! ## String-concatenation policy
//! PTX bodies MUST NOT use `format!()` for segments containing `%r`, `%rd`, `%f`, `%fd`,
//! etc. — Rust 2024 treats those as format argument placeholders. All register references
//! live in string literals using `\` line-continuation, concatenated with `hdr + body`.

/// Build a PTX file header for the given SM version.
fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

/// Encode a `f32` constant as a PTX hex literal (`0Fxxxxxxxx`).
#[allow(dead_code)]
fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: fitness_eval ──────────────────────────────────────────────────

/// Sphere fitness evaluation kernel — each thread computes the sum-of-squares fitness
/// for one individual.
///
/// Signature: `fitness_eval_kernel(x: *f64, fitness: *f64, n_dims: u32, pop_size: u32)`
#[must_use]
pub fn fitness_eval_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// fitness_eval_kernel: sphere (sum of squares) for each individual\n\
.visible .entry fitness_eval_kernel(\n\
    .param .u64 p_x,\n\
    .param .u64 p_fitness,\n\
    .param .u32 p_n_dims,\n\
    .param .u32 p_pop_size\n\
)\n\
{\n\
    .reg .u64  %rd<16>;\n\
    .reg .u32  %r<16>;\n\
    .reg .f64  %fd<8>;\n\
    .reg .pred %p0;\n\
\n\
    ld.param.u64  %rd0, [p_x];\n\
    ld.param.u64  %rd1, [p_fitness];\n\
    ld.param.u32  %r0,  [p_n_dims];\n\
    ld.param.u32  %r1,  [p_pop_size];\n\
\n\
    mov.u32       %r2, %ntid.x;\n\
    mov.u32       %r3, %ctaid.x;\n\
    mov.u32       %r4, %tid.x;\n\
    mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    setp.ge.u32   %p0, %r5, %r1;\n\
    @%p0 bra $FE_DONE;\n\
\n\
    // base offset for this individual: r5 * n_dims * 8 bytes\n\
    mul.lo.u32    %r6, %r5, %r0;\n\
    cvt.u64.u32   %rd2, %r6;\n\
    shl.b64       %rd3, %rd2, 3;\n\
    add.u64       %rd4, %rd0, %rd3;\n\
\n\
    // accumulate sum of squares over n_dims elements\n\
    mov.f64       %fd0, 0d0000000000000000;\n\
    mov.u32       %r7, 0;\n\
$FE_LOOP:\n\
    setp.ge.u32   %p0, %r7, %r0;\n\
    @%p0 bra $FE_WRITE;\n\
    cvt.u64.u32   %rd5, %r7;\n\
    shl.b64       %rd6, %rd5, 3;\n\
    add.u64       %rd7, %rd4, %rd6;\n\
    ld.global.f64 %fd1, [%rd7];\n\
    fma.rn.f64    %fd0, %fd1, %fd1, %fd0;\n\
    add.u32       %r7, %r7, 1;\n\
    bra $FE_LOOP;\n\
$FE_WRITE:\n\
    cvt.u64.u32   %rd8, %r5;\n\
    shl.b64       %rd9, %rd8, 3;\n\
    add.u64       %rd10, %rd1, %rd9;\n\
    st.global.f64 [%rd10], %fd0;\n\
$FE_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

// ─── Kernel 2: tournament_select ─────────────────────────────────────────────

/// k=2 tournament selection kernel.
///
/// Signature: `tournament_select_kernel(fitness: *f64, selected_idx: *u32, pop_size: u32, n_select: u32, seed: u64)`
#[must_use]
pub fn tournament_select_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// tournament_select_kernel: k=2 tournament, pick lower-fitness winner\n\
.visible .entry tournament_select_kernel(\n\
    .param .u64 p_fitness,\n\
    .param .u64 p_selected_idx,\n\
    .param .u32 p_pop_size,\n\
    .param .u32 p_n_select,\n\
    .param .u64 p_seed\n\
)\n\
{\n\
    .reg .u64  %rd<16>;\n\
    .reg .u32  %r<16>;\n\
    .reg .f64  %fd<8>;\n\
    .reg .pred %p0, %p1;\n\
\n\
    ld.param.u64  %rd0, [p_fitness];\n\
    ld.param.u64  %rd1, [p_selected_idx];\n\
    ld.param.u32  %r0,  [p_pop_size];\n\
    ld.param.u32  %r1,  [p_n_select];\n\
    ld.param.u64  %rd2, [p_seed];\n\
\n\
    mov.u32       %r2, %ntid.x;\n\
    mov.u32       %r3, %ctaid.x;\n\
    mov.u32       %r4, %tid.x;\n\
    mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    setp.ge.u32   %p0, %r5, %r1;\n\
    @%p0 bra $TS_DONE;\n\
\n\
    // Simple LCG per thread: state = (seed ^ tid) * MUL + ADD\n\
    cvt.u64.u32   %rd3, %r5;\n\
    xor.b64       %rd4, %rd2, %rd3;\n\
    // candidate a\n\
    mov.u64       %rd5, 6364136223846793005;\n\
    mul.lo.u64    %rd6, %rd4, %rd5;\n\
    add.u64       %rd6, %rd6, 1442695040888963407;\n\
    cvt.u32.u64   %r6, %rd6;\n\
    rem.u32       %r7, %r6, %r0;\n\
    // candidate b\n\
    mul.lo.u64    %rd7, %rd6, %rd5;\n\
    add.u64       %rd7, %rd7, 1442695040888963407;\n\
    cvt.u32.u64   %r8, %rd7;\n\
    rem.u32       %r9, %r8, %r0;\n\
\n\
    // load fitness[r7] and fitness[r9]\n\
    cvt.u64.u32   %rd8, %r7;\n\
    shl.b64       %rd8, %rd8, 3;\n\
    add.u64       %rd8, %rd0, %rd8;\n\
    ld.global.f64 %fd0, [%rd8];\n\
    cvt.u64.u32   %rd9, %r9;\n\
    shl.b64       %rd9, %rd9, 3;\n\
    add.u64       %rd9, %rd0, %rd9;\n\
    ld.global.f64 %fd1, [%rd9];\n\
\n\
    // winner = lower fitness\n\
    setp.le.f64   %p1, %fd0, %fd1;\n\
    @%p1 mov.u32  %r10, %r7;\n\
    @!%p1 mov.u32 %r10, %r9;\n\
\n\
    // store winner index\n\
    cvt.u64.u32   %rd10, %r5;\n\
    shl.b64       %rd10, %rd10, 2;\n\
    add.u64       %rd10, %rd1, %rd10;\n\
    st.global.u32 [%rd10], %r10;\n\
$TS_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

// ─── Kernel 3: gaussian_mutate ────────────────────────────────────────────────

/// Per-gene Gaussian mutation with probability p_mut.
///
/// Each surviving gene receives a true normal perturbation `delta = sigma * z`,
/// where `z ~ N(0,1)` is produced by the **Box–Muller transform**
/// `z = sqrt(-2·ln u1) · cos(2π·u2)` from two independent uniforms `u1, u2 ∈ [0,1)`.
///
/// PTX has no `sin.approx`/`cos.approx`/`lg2.approx` for `.f64`, so the
/// transcendentals are evaluated in software:
///   * `ln u1 = lg2.approx.f32(u1) · ln 2` (`u1` clamped to `2^-53` to avoid
///     `ln(0)`; the log base-2 is taken in `f32`, whose ~2^-23 relative error is
///     ample for mutation noise, then widened back to `f64`),
///   * `cos(a)` via Cody–Waite octant reduction `k = round(a·2/π)`,
///     `x = a − k·(π/2) ∈ [-π/4, π/4]`, evaluating degree-10 cos / degree-11 sin
///     Taylor series on the reduced argument and selecting the result from
///     `{cos x, −sin x, −cos x, sin x}` by `k mod 4`. Octant reduction keeps the
///     polynomial argument tiny, giving < 1.2e-10 absolute error over `[0,2π)` —
///     far tighter than a naive `[-π,π]` series.
///
/// Signature: `gaussian_mutate_kernel(genome: *f64, n: u32, sigma: f64, p_mut: f64, seed: u64)`
#[must_use]
pub fn gaussian_mutate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// gaussian_mutate_kernel: add N(0,sigma) to each gene with prob p_mut\n\
// delta = sigma * BoxMuller(u1,u2),  z = sqrt(-2 ln u1) * cos(2 pi u2)\n\
.visible .entry gaussian_mutate_kernel(\n\
    .param .u64 p_genome,\n\
    .param .u32 p_n,\n\
    .param .f64 p_sigma,\n\
    .param .f64 p_p_mut,\n\
    .param .u64 p_seed\n\
)\n\
{\n\
    .reg .u64  %rd<16>;\n\
    .reg .u32  %r<8>;\n\
    .reg .f32  %f<4>;\n\
    .reg .f64  %fd<48>;\n\
    .reg .s64  %sd<4>;\n\
    .reg .pred %p0, %p1, %p2, %p3;\n\
\n\
    ld.param.u64  %rd0, [p_genome];\n\
    ld.param.u32  %r0,  [p_n];\n\
    ld.param.f64  %fd0, [p_sigma];\n\
    ld.param.f64  %fd1, [p_p_mut];\n\
    ld.param.u64  %rd1, [p_seed];\n\
\n\
    mov.u32       %r1, %ntid.x;\n\
    mov.u32       %r2, %ctaid.x;\n\
    mov.u32       %r3, %tid.x;\n\
    mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    setp.ge.u32   %p0, %r4, %r0;\n\
    @%p0 bra $GM_DONE;\n\
\n\
    // LCG: state = (seed XOR tid) * MUL + ADD; advance three times for\n\
    // three independent uniforms: u_gate (mutation test), u1, u2 (Box-Muller).\n\
    cvt.u64.u32   %rd2, %r4;\n\
    xor.b64       %rd3, %rd1, %rd2;\n\
    mov.u64       %rd4, 6364136223846793005;\n\
    mov.f64       %fd3, 0d3CA0000000000000;\n\
    // draw 1: u_gate\n\
    mul.lo.u64    %rd5, %rd3, %rd4;\n\
    add.u64       %rd5, %rd5, 1442695040888963407;\n\
    shr.u64       %rd6, %rd5, 11;\n\
    cvt.rn.f64.u64 %fd2, %rd6;\n\
    mul.rn.f64    %fd2, %fd2, %fd3;\n\
    setp.ge.f64   %p1, %fd2, %fd1;\n\
    @%p1 bra $GM_DONE;\n\
\n\
    // draw 2: u1 in [0,1)\n\
    mul.lo.u64    %rd5, %rd5, %rd4;\n\
    add.u64       %rd5, %rd5, 1442695040888963407;\n\
    shr.u64       %rd7, %rd5, 11;\n\
    cvt.rn.f64.u64 %fd4, %rd7;\n\
    mul.rn.f64    %fd4, %fd4, %fd3;\n\
    // draw 3: u2 in [0,1)\n\
    mul.lo.u64    %rd5, %rd5, %rd4;\n\
    add.u64       %rd5, %rd5, 1442695040888963407;\n\
    shr.u64       %rd8, %rd5, 11;\n\
    cvt.rn.f64.u64 %fd5, %rd8;\n\
    mul.rn.f64    %fd5, %fd5, %fd3;\n\
\n\
    // --- Box-Muller radius: r = sqrt(-2 * ln u1) ------------------------------\n\
    // clamp u1 away from 0 so ln(u1) is finite (eps = 2^-53)\n\
    max.f64       %fd6, %fd4, %fd3;\n\
    // ln(u1) = lg2(u1) * ln(2). PTX has no lg2.approx.f64, so take the log\n\
    // base-2 in f32 (ample precision for mutation noise) then widen to f64.\n\
    cvt.rn.f32.f64 %f0, %fd6;\n\
    lg2.approx.f32 %f1, %f0;\n\
    cvt.f64.f32   %fd7, %f1;\n\
    mov.f64       %fd8, 0d3FE62E42FEFA39EF;\n\
    mul.rn.f64    %fd9, %fd7, %fd8;\n\
    // -2 * ln(u1)\n\
    mov.f64       %fd10, 0dC000000000000000;\n\
    mul.rn.f64    %fd11, %fd10, %fd9;\n\
    sqrt.rn.f64   %fd12, %fd11;\n\
\n\
    // --- angle a = 2*pi*u2 -----------------------------------------------------\n\
    mov.f64       %fd13, 0d401921FB54442D18;\n\
    mul.rn.f64    %fd14, %fd13, %fd5;\n\
    // octant reduction: k = round(a * 2/pi); x = a - k*(pi/2), x in [-pi/4,pi/4]\n\
    mov.f64       %fd15, 0d3FE45F306DC9C883;\n\
    mul.rn.f64    %fd16, %fd14, %fd15;\n\
    cvt.rni.s64.f64 %sd0, %fd16;\n\
    cvt.rn.f64.s64 %fd17, %sd0;\n\
    mov.f64       %fd18, 0d3FF921FB54442D18;\n\
    mul.rn.f64    %fd19, %fd17, %fd18;\n\
    sub.rn.f64    %fd20, %fd14, %fd19;\n\
\n\
    // u = x^2, x3 = x^3\n\
    mul.rn.f64    %fd21, %fd20, %fd20;\n\
    mul.rn.f64    %fd22, %fd21, %fd20;\n\
\n\
    // cos(x) on [-pi/4,pi/4]: Horner in u\n\
    //   1 - u/2 + u^2/24 - u^3/720 + u^4/40320 - u^5/3628800\n\
    mov.f64       %fd23, 0dBE927E4FB7789F5C;\n\
    mov.f64       %fd24, 0d3EFA01A01A01A01A;\n\
    fma.rn.f64    %fd25, %fd23, %fd21, %fd24;\n\
    mov.f64       %fd26, 0dBF56C16C16C16C17;\n\
    fma.rn.f64    %fd25, %fd25, %fd21, %fd26;\n\
    mov.f64       %fd27, 0d3FA5555555555555;\n\
    fma.rn.f64    %fd25, %fd25, %fd21, %fd27;\n\
    mov.f64       %fd28, 0dBFE0000000000000;\n\
    fma.rn.f64    %fd25, %fd25, %fd21, %fd28;\n\
    mov.f64       %fd29, 0d3FF0000000000000;\n\
    fma.rn.f64    %fd30, %fd25, %fd21, %fd29;\n\
\n\
    // sin(x) on [-pi/4,pi/4]: x + x^3 * P(u)\n\
    //   P(u) = -1/6 + u/120 - u^2/5040 + u^3/362880 - u^4/39916800\n\
    mov.f64       %fd31, 0dBE5AE64567F544E4;\n\
    mov.f64       %fd32, 0d3EC71DE3A556C734;\n\
    fma.rn.f64    %fd33, %fd31, %fd21, %fd32;\n\
    mov.f64       %fd34, 0dBF2A01A01A01A01A;\n\
    fma.rn.f64    %fd33, %fd33, %fd21, %fd34;\n\
    mov.f64       %fd35, 0d3F81111111111111;\n\
    fma.rn.f64    %fd33, %fd33, %fd21, %fd35;\n\
    mov.f64       %fd36, 0dBFC5555555555555;\n\
    fma.rn.f64    %fd33, %fd33, %fd21, %fd36;\n\
    fma.rn.f64    %fd37, %fd33, %fd22, %fd20;\n\
\n\
    // quadrant select cos(a) from {cos x, -sin x, -cos x, sin x} by (k mod 4)\n\
    and.b64       %sd1, %sd0, 3;\n\
    // use_sin = (k & 1) == 1  -> pick sin x else cos x\n\
    and.b64       %sd2, %sd0, 1;\n\
    setp.eq.s64   %p2, %sd2, 1;\n\
    selp.f64      %fd38, %fd37, %fd30, %p2;\n\
    // negate = ((k mod 4) ^ ((k mod 4) >> 1)) & 1  -> true for q in {1,2}\n\
    shr.s64       %sd3, %sd1, 1;\n\
    xor.b64       %sd3, %sd3, %sd1;\n\
    and.b64       %sd3, %sd3, 1;\n\
    setp.eq.s64   %p3, %sd3, 1;\n\
    neg.f64       %fd39, %fd38;\n\
    selp.f64      %fd40, %fd39, %fd38, %p3;\n\
\n\
    // z0 = r * cos(a);  delta = sigma * z0\n\
    mul.rn.f64    %fd41, %fd12, %fd40;\n\
    mul.rn.f64    %fd42, %fd0, %fd41;\n\
\n\
    // load gene, add delta, store\n\
    cvt.u64.u32   %rd9, %r4;\n\
    shl.b64       %rd9, %rd9, 3;\n\
    add.u64       %rd9, %rd0, %rd9;\n\
    ld.global.f64 %fd2, [%rd9];\n\
    add.rn.f64    %fd2, %fd2, %fd42;\n\
    st.global.f64 [%rd9], %fd2;\n\
$GM_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

// ─── Kernel 4: nsga_crowding ──────────────────────────────────────────────────

/// Crowding distance contribution kernel for one objective (sorted order).
///
/// Signature: `nsga_crowding_kernel(sorted_obj: *f64, crowd_dist: *f64, n: u32, obj_range: f64)`
#[must_use]
pub fn nsga_crowding_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// nsga_crowding_kernel: accumulate crowding distance for sorted objective values\n\
.visible .entry nsga_crowding_kernel(\n\
    .param .u64 p_sorted_obj,\n\
    .param .u64 p_crowd_dist,\n\
    .param .u32 p_n,\n\
    .param .f64 p_obj_range\n\
)\n\
{\n\
    .reg .u64  %rd<16>;\n\
    .reg .u32  %r<8>;\n\
    .reg .f64  %fd<8>;\n\
    .reg .pred %p0, %p1, %p2;\n\
\n\
    ld.param.u64  %rd0, [p_sorted_obj];\n\
    ld.param.u64  %rd1, [p_crowd_dist];\n\
    ld.param.u32  %r0,  [p_n];\n\
    ld.param.f64  %fd0, [p_obj_range];\n\
\n\
    mov.u32       %r1, %ntid.x;\n\
    mov.u32       %r2, %ctaid.x;\n\
    mov.u32       %r3, %tid.x;\n\
    mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    // boundaries get infinity crowding distance — skip them\n\
    setp.eq.u32   %p0, %r4, 0;\n\
    @%p0 bra $NC_DONE;\n\
    sub.u32       %r5, %r0, 1;\n\
    setp.ge.u32   %p1, %r4, %r5;\n\
    @%p1 bra $NC_DONE;\n\
    setp.ge.u32   %p2, %r4, %r0;\n\
    @%p2 bra $NC_DONE;\n\
\n\
    // load sorted_obj[i-1], sorted_obj[i+1]\n\
    sub.u32       %r6, %r4, 1;\n\
    cvt.u64.u32   %rd2, %r6;\n\
    shl.b64       %rd2, %rd2, 3;\n\
    add.u64       %rd2, %rd0, %rd2;\n\
    ld.global.f64 %fd1, [%rd2];\n\
    add.u32       %r7, %r4, 1;\n\
    cvt.u64.u32   %rd3, %r7;\n\
    shl.b64       %rd3, %rd3, 3;\n\
    add.u64       %rd3, %rd0, %rd3;\n\
    ld.global.f64 %fd2, [%rd3];\n\
\n\
    // contribution = (obj[i+1] - obj[i-1]) / obj_range\n\
    sub.rn.f64    %fd3, %fd2, %fd1;\n\
    div.rn.f64    %fd3, %fd3, %fd0;\n\
\n\
    // atomic add to crowd_dist[i]\n\
    cvt.u64.u32   %rd4, %r4;\n\
    shl.b64       %rd4, %rd4, 3;\n\
    add.u64       %rd4, %rd1, %rd4;\n\
    atom.global.add.f64 %fd4, [%rd4], %fd3;\n\
$NC_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

// ─── Kernel 5: pso_update ────────────────────────────────────────────────────

/// PSO velocity and position update kernel.
///
/// Signature: `pso_update_kernel(pos: *f64, vel: *f64, pbest: *f64, gbest: *f64, n: u32, w_inertia: f64, c1: f64, c2: f64, seed: u64)`
#[must_use]
pub fn pso_update_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// pso_update_kernel: v <- w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x); x <- x+v\n\
.visible .entry pso_update_kernel(\n\
    .param .u64 p_pos,\n\
    .param .u64 p_vel,\n\
    .param .u64 p_pbest,\n\
    .param .u64 p_gbest,\n\
    .param .u32 p_n,\n\
    .param .f64 p_w,\n\
    .param .f64 p_c1,\n\
    .param .f64 p_c2,\n\
    .param .u64 p_seed\n\
)\n\
{\n\
    .reg .u64  %rd<20>;\n\
    .reg .u32  %r<8>;\n\
    .reg .f64  %fd<16>;\n\
    .reg .pred %p0;\n\
\n\
    ld.param.u64  %rd0,  [p_pos];\n\
    ld.param.u64  %rd1,  [p_vel];\n\
    ld.param.u64  %rd2,  [p_pbest];\n\
    ld.param.u64  %rd3,  [p_gbest];\n\
    ld.param.u32  %r0,   [p_n];\n\
    ld.param.f64  %fd0,  [p_w];\n\
    ld.param.f64  %fd1,  [p_c1];\n\
    ld.param.f64  %fd2,  [p_c2];\n\
    ld.param.u64  %rd4,  [p_seed];\n\
\n\
    mov.u32       %r1, %ntid.x;\n\
    mov.u32       %r2, %ctaid.x;\n\
    mov.u32       %r3, %tid.x;\n\
    mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    setp.ge.u32   %p0, %r4, %r0;\n\
    @%p0 bra $PSO_DONE;\n\
\n\
    cvt.u64.u32   %rd5, %r4;\n\
    shl.b64       %rd5, %rd5, 3;\n\
    add.u64       %rd6,  %rd0, %rd5;\n\
    add.u64       %rd7,  %rd1, %rd5;\n\
    add.u64       %rd8,  %rd2, %rd5;\n\
    // gbest is shared; use dim index as offset into gbest\n\
    add.u64       %rd9,  %rd3, %rd5;\n\
\n\
    ld.global.f64 %fd3, [%rd6];\n\
    ld.global.f64 %fd4, [%rd7];\n\
    ld.global.f64 %fd5, [%rd8];\n\
    ld.global.f64 %fd6, [%rd9];\n\
\n\
    // LCG for r1\n\
    cvt.u64.u32   %rd10, %r4;\n\
    xor.b64       %rd10, %rd10, %rd4;\n\
    mov.u64       %rd11, 6364136223846793005;\n\
    mul.lo.u64    %rd10, %rd10, %rd11;\n\
    add.u64       %rd10, %rd10, 1442695040888963407;\n\
    shr.u64       %rd12, %rd10, 11;\n\
    cvt.rn.f64.u64 %fd7, %rd12;\n\
    mov.f64       %fd8, 0d3CA0000000000000;\n\
    mul.rn.f64    %fd7, %fd7, %fd8;\n\
    // LCG for r2\n\
    mul.lo.u64    %rd10, %rd10, %rd11;\n\
    add.u64       %rd10, %rd10, 1442695040888963407;\n\
    shr.u64       %rd12, %rd10, 11;\n\
    cvt.rn.f64.u64 %fd9, %rd12;\n\
    mul.rn.f64    %fd9, %fd9, %fd8;\n\
\n\
    // v_new = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)\n\
    mul.rn.f64    %fd10, %fd0, %fd4;\n\
    sub.rn.f64    %fd11, %fd5, %fd3;\n\
    mul.rn.f64    %fd11, %fd1, %fd11;\n\
    mul.rn.f64    %fd11, %fd7, %fd11;\n\
    sub.rn.f64    %fd12, %fd6, %fd3;\n\
    mul.rn.f64    %fd12, %fd2, %fd12;\n\
    mul.rn.f64    %fd12, %fd9, %fd12;\n\
    add.rn.f64    %fd10, %fd10, %fd11;\n\
    add.rn.f64    %fd10, %fd10, %fd12;\n\
    // x_new = x + v_new\n\
    add.rn.f64    %fd13, %fd3, %fd10;\n\
    st.global.f64 [%rd7], %fd10;\n\
    st.global.f64 [%rd6], %fd13;\n\
$PSO_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

// ─── Kernel 6: de_mutate ─────────────────────────────────────────────────────

/// DE/rand/1 mutation kernel: `v = a + F*(b - c)`.
///
/// Signature: `de_mutate_kernel(pop: *f64, mutant: *f64, n_dims: u32, pop_size: u32, f_scale: f64, target_idx: u32, seed: u64)`
#[must_use]
pub fn de_mutate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// de_mutate_kernel: DE/rand/1: mutant = pop[r1] + F*(pop[r2]-pop[r3])\n\
.visible .entry de_mutate_kernel(\n\
    .param .u64 p_pop,\n\
    .param .u64 p_mutant,\n\
    .param .u32 p_n_dims,\n\
    .param .u32 p_pop_size,\n\
    .param .f64 p_f_scale,\n\
    .param .u32 p_target_idx,\n\
    .param .u64 p_seed\n\
)\n\
{\n\
    .reg .u64  %rd<20>;\n\
    .reg .u32  %r<16>;\n\
    .reg .f64  %fd<8>;\n\
    .reg .pred %p0;\n\
\n\
    ld.param.u64  %rd0,  [p_pop];\n\
    ld.param.u64  %rd1,  [p_mutant];\n\
    ld.param.u32  %r0,   [p_n_dims];\n\
    ld.param.u32  %r1,   [p_pop_size];\n\
    ld.param.f64  %fd0,  [p_f_scale];\n\
    ld.param.u32  %r2,   [p_target_idx];\n\
    ld.param.u64  %rd2,  [p_seed];\n\
\n\
    mov.u32       %r3, %ntid.x;\n\
    mov.u32       %r4, %ctaid.x;\n\
    mov.u32       %r5, %tid.x;\n\
    mad.lo.u32    %r6, %r3, %r4, %r5;\n\
    setp.ge.u32   %p0, %r6, %r0;\n\
    @%p0 bra $DE_DONE;\n\
\n\
    // LCG to pick r1, r2, r3 (distinct from target and each other; simplified here)\n\
    cvt.u64.u32   %rd3, %r6;\n\
    xor.b64       %rd3, %rd3, %rd2;\n\
    mov.u64       %rd4, 6364136223846793005;\n\
    mul.lo.u64    %rd5, %rd3, %rd4;\n\
    add.u64       %rd5, %rd5, 1442695040888963407;\n\
    cvt.u32.u64   %r7, %rd5;\n\
    rem.u32       %r8, %r7, %r1;\n\
    // ensure r8 != target\n\
    setp.eq.u32   %p0, %r8, %r2;\n\
    @%p0 add.u32  %r8, %r8, 1;\n\
    rem.u32       %r8, %r8, %r1;\n\
    // r2_idx\n\
    mul.lo.u64    %rd5, %rd5, %rd4;\n\
    add.u64       %rd5, %rd5, 1442695040888963407;\n\
    cvt.u32.u64   %r9, %rd5;\n\
    rem.u32       %r9, %r9, %r1;\n\
    setp.eq.u32   %p0, %r9, %r2;\n\
    @%p0 add.u32  %r9, %r9, 1;\n\
    rem.u32       %r9, %r9, %r1;\n\
    // r3_idx\n\
    mul.lo.u64    %rd5, %rd5, %rd4;\n\
    add.u64       %rd5, %rd5, 1442695040888963407;\n\
    cvt.u32.u64   %r10, %rd5;\n\
    rem.u32       %r10, %r10, %r1;\n\
    setp.eq.u32   %p0, %r10, %r2;\n\
    @%p0 add.u32  %r10, %r10, 1;\n\
    rem.u32       %r10, %r10, %r1;\n\
\n\
    // compute byte offsets for dim %r6 in each individual\n\
    cvt.u64.u32   %rd6, %r6;\n\
    shl.b64       %rd6, %rd6, 3;\n\
    mul.lo.u32    %r11, %r8, %r0;\n\
    cvt.u64.u32   %rd7, %r11;\n\
    shl.b64       %rd7, %rd7, 3;\n\
    add.u64       %rd7, %rd0, %rd7;\n\
    add.u64       %rd7, %rd7, %rd6;\n\
    mul.lo.u32    %r12, %r9, %r0;\n\
    cvt.u64.u32   %rd8, %r12;\n\
    shl.b64       %rd8, %rd8, 3;\n\
    add.u64       %rd8, %rd0, %rd8;\n\
    add.u64       %rd8, %rd8, %rd6;\n\
    mul.lo.u32    %r13, %r10, %r0;\n\
    cvt.u64.u32   %rd9, %r13;\n\
    shl.b64       %rd9, %rd9, 3;\n\
    add.u64       %rd9, %rd0, %rd9;\n\
    add.u64       %rd9, %rd9, %rd6;\n\
\n\
    ld.global.f64 %fd1, [%rd7];\n\
    ld.global.f64 %fd2, [%rd8];\n\
    ld.global.f64 %fd3, [%rd9];\n\
    sub.rn.f64    %fd4, %fd2, %fd3;\n\
    fma.rn.f64    %fd5, %fd0, %fd4, %fd1;\n\
\n\
    // store to mutant[target_idx * n_dims + dim]\n\
    mul.lo.u32    %r14, %r2, %r0;\n\
    cvt.u64.u32   %rd10, %r14;\n\
    shl.b64       %rd10, %rd10, 3;\n\
    add.u64       %rd10, %rd1, %rd10;\n\
    add.u64       %rd10, %rd10, %rd6;\n\
    st.global.f64 [%rd10], %fd5;\n\
$DE_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

// ─── Kernel 7: cmaes_sample ───────────────────────────────────────────────────

/// CMA-ES sample kernel: `x = m + sigma * B * D * z`, where `z ~ N(0,I)`.
///
/// Signature: `cmaes_sample_kernel(m: *f64, sigma: f64, B: *f64, D: *f64, z: *f64, x: *f64, n_dims: u32, pop_size: u32)`
#[must_use]
pub fn cmaes_sample_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// cmaes_sample_kernel: x_k = m + sigma * B * D * z_k\n\
// Thread (k, i) computes dimension i of sample k\n\
// Launched as (pop_size * n_dims) threads\n\
.visible .entry cmaes_sample_kernel(\n\
    .param .u64 p_m,\n\
    .param .f64 p_sigma,\n\
    .param .u64 p_B,\n\
    .param .u64 p_D,\n\
    .param .u64 p_z,\n\
    .param .u64 p_x,\n\
    .param .u32 p_n_dims,\n\
    .param .u32 p_pop_size\n\
)\n\
{\n\
    .reg .u64  %rd<20>;\n\
    .reg .u32  %r<12>;\n\
    .reg .f64  %fd<10>;\n\
    .reg .pred %p0;\n\
\n\
    ld.param.u64  %rd0, [p_m];\n\
    ld.param.f64  %fd0, [p_sigma];\n\
    ld.param.u64  %rd1, [p_B];\n\
    ld.param.u64  %rd2, [p_D];\n\
    ld.param.u64  %rd3, [p_z];\n\
    ld.param.u64  %rd4, [p_x];\n\
    ld.param.u32  %r0,  [p_n_dims];\n\
    ld.param.u32  %r1,  [p_pop_size];\n\
\n\
    mov.u32       %r2, %ntid.x;\n\
    mov.u32       %r3, %ctaid.x;\n\
    mov.u32       %r4, %tid.x;\n\
    mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    // total threads = pop_size * n_dims\n\
    mul.lo.u32    %r6, %r1, %r0;\n\
    setp.ge.u32   %p0, %r5, %r6;\n\
    @%p0 bra $CS_DONE;\n\
\n\
    // sample k = r5 / n_dims, dim i = r5 % n_dims\n\
    div.u32       %r7, %r5, %r0;\n\
    rem.u32       %r8, %r5, %r0;\n\
\n\
    // compute (B * D * z)[k][i] = sum_j B[i,j] * D[j] * z[k][j]\n\
    mov.f64       %fd1, 0d0000000000000000;\n\
    mov.u32       %r9, 0;\n\
$CS_LOOP:\n\
    setp.ge.u32   %p0, %r9, %r0;\n\
    @%p0 bra $CS_WRITE;\n\
    // B[i,j] = B[i * n_dims + j]\n\
    mul.lo.u32    %r10, %r8, %r0;\n\
    add.u32       %r10, %r10, %r9;\n\
    cvt.u64.u32   %rd5, %r10;\n\
    shl.b64       %rd5, %rd5, 3;\n\
    add.u64       %rd6, %rd1, %rd5;\n\
    ld.global.f64 %fd2, [%rd6];\n\
    // D[j]\n\
    cvt.u64.u32   %rd7, %r9;\n\
    shl.b64       %rd7, %rd7, 3;\n\
    add.u64       %rd8, %rd2, %rd7;\n\
    ld.global.f64 %fd3, [%rd8];\n\
    // z[k][j] = z[k * n_dims + j]\n\
    mul.lo.u32    %r10, %r7, %r0;\n\
    add.u32       %r10, %r10, %r9;\n\
    cvt.u64.u32   %rd9, %r10;\n\
    shl.b64       %rd9, %rd9, 3;\n\
    add.u64       %rd10, %rd3, %rd9;\n\
    ld.global.f64 %fd4, [%rd10];\n\
    mul.rn.f64    %fd5, %fd3, %fd4;\n\
    fma.rn.f64    %fd1, %fd2, %fd5, %fd1;\n\
    add.u32       %r9, %r9, 1;\n\
    bra $CS_LOOP;\n\
$CS_WRITE:\n\
    // x[k][i] = m[i] + sigma * (B*D*z)[k][i]\n\
    cvt.u64.u32   %rd11, %r8;\n\
    shl.b64       %rd11, %rd11, 3;\n\
    add.u64       %rd12, %rd0, %rd11;\n\
    ld.global.f64 %fd6, [%rd12];\n\
    mul.rn.f64    %fd7, %fd0, %fd1;\n\
    add.rn.f64    %fd8, %fd6, %fd7;\n\
    cvt.u64.u32   %rd13, %r5;\n\
    shl.b64       %rd13, %rd13, 3;\n\
    add.u64       %rd14, %rd4, %rd13;\n\
    st.global.f64 [%rd14], %fd8;\n\
$CS_DONE:\n\
    ret;\n\
}\n";
    hdr + body
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SM versions spanning Turing through Blackwell.
    const ALL_SM: &[u32] = &[75, 80, 86, 90, 100, 120];

    /// Verify the header carries the expected target/address-size for `sm`.
    fn check_header(ptx: &str, sm: u32) {
        assert!(
            ptx.contains(&format!(".target sm_{sm}")),
            "missing .target sm_{sm}"
        );
        assert!(ptx.contains(".address_size 64"), "missing .address_size 64");
    }

    #[test]
    fn fitness_eval_all_sm() {
        for &sm in ALL_SM {
            check_header(&fitness_eval_ptx(sm), sm);
        }
    }

    #[test]
    fn tournament_select_all_sm() {
        for &sm in ALL_SM {
            check_header(&tournament_select_ptx(sm), sm);
        }
    }

    #[test]
    fn gaussian_mutate_all_sm() {
        for &sm in ALL_SM {
            check_header(&gaussian_mutate_ptx(sm), sm);
        }
    }

    #[test]
    fn nsga_crowding_all_sm() {
        for &sm in ALL_SM {
            check_header(&nsga_crowding_ptx(sm), sm);
        }
    }

    #[test]
    fn pso_update_all_sm() {
        for &sm in ALL_SM {
            check_header(&pso_update_ptx(sm), sm);
        }
    }

    #[test]
    fn de_mutate_all_sm() {
        for &sm in ALL_SM {
            check_header(&de_mutate_ptx(sm), sm);
        }
    }

    #[test]
    fn cmaes_sample_all_sm() {
        for &sm in ALL_SM {
            check_header(&cmaes_sample_ptx(sm), sm);
        }
    }

    #[test]
    fn gaussian_mutate_emits_box_muller_sequence() {
        // The mutation kernel must perform a real Box-Muller transform:
        //   r = sqrt(-2 * ln u1),  z = r * cos(2 pi u2),  delta = sigma * z.
        let ptx = gaussian_mutate_ptx(80);
        // ln u1 via lg2.approx.f32 then widen (PTX has no lg2.approx.f64).
        assert!(
            ptx.contains("lg2.approx.f32"),
            "Box-Muller log step missing"
        );
        // radius via sqrt of -2 ln u1.
        assert!(ptx.contains("sqrt.rn.f64"), "Box-Muller sqrt step missing");
        // cosine of the reduced angle via octant reduction (cvt.rni + selp).
        assert!(
            ptx.contains("cvt.rni.s64.f64"),
            "trig range-reduction missing"
        );
        assert!(ptx.contains("selp.f64"), "trig quadrant selection missing");
        // The 2*pi angle constant must be present (0d401921FB54442D18).
        assert!(
            ptx.contains("0d401921FB54442D18"),
            "2*pi angle constant missing"
        );
    }

    #[test]
    fn gaussian_mutate_no_longer_uses_uniform_proxy() {
        // Regression: the old kernel approximated the normal noise with a bare
        // uniform draw and admitted so in a comment. Neither the stale comment
        // nor a single-uniform delta should remain.
        let ptx = gaussian_mutate_ptx(80);
        assert!(
            !ptx.contains("approximation"),
            "stale 'approximation' comment still present"
        );
        assert!(
            !ptx.contains("placeholder"),
            "stale 'placeholder' comment still present"
        );
        assert!(
            ptx.contains("BoxMuller"),
            "kernel should document the Box-Muller transform"
        );
    }

    #[test]
    fn gaussian_mutate_draws_three_independent_uniforms() {
        // Three LCG advances: u_gate (mutation test), u1 and u2 (Box-Muller).
        let ptx = gaussian_mutate_ptx(80);
        let advances = ptx.matches("1442695040888963407").count();
        assert!(
            advances >= 3,
            "expected >= 3 LCG advances for three uniforms, found {advances}"
        );
    }

    #[test]
    fn blackwell_uses_ptx_87() {
        assert!(gaussian_mutate_ptx(120).contains(".version 8.7"));
    }

    #[test]
    fn turing_uses_ptx_75() {
        assert!(gaussian_mutate_ptx(75).contains(".version 7.5"));
    }
}
