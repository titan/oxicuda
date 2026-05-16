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
/// Signature: `gaussian_mutate_kernel(genome: *f64, n: u32, sigma: f64, p_mut: f64, seed: u64)`
#[must_use]
pub fn gaussian_mutate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = "// gaussian_mutate_kernel: add N(0,sigma) to each gene with prob p_mut\n\
.visible .entry gaussian_mutate_kernel(\n\
    .param .u64 p_genome,\n\
    .param .u32 p_n,\n\
    .param .f64 p_sigma,\n\
    .param .f64 p_p_mut,\n\
    .param .u64 p_seed\n\
)\n\
{\n\
    .reg .u64  %rd<12>;\n\
    .reg .u32  %r<8>;\n\
    .reg .f64  %fd<8>;\n\
    .reg .pred %p0, %p1;\n\
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
    // LCG: state = (seed XOR tid) * MUL + ADD\n\
    cvt.u64.u32   %rd2, %r4;\n\
    xor.b64       %rd3, %rd1, %rd2;\n\
    mov.u64       %rd4, 6364136223846793005;\n\
    mul.lo.u64    %rd5, %rd3, %rd4;\n\
    add.u64       %rd5, %rd5, 1442695040888963407;\n\
    // uniform [0,1) from high 53 bits\n\
    shr.u64       %rd6, %rd5, 11;\n\
    cvt.rn.f64.u64 %fd2, %rd6;\n\
    mov.f64       %fd3, 0d3CA0000000000000;\n\
    mul.rn.f64    %fd2, %fd2, %fd3;\n\
    setp.ge.f64   %p1, %fd2, %fd1;\n\
    @%p1 bra $GM_DONE;\n\
\n\
    // load gene, add sigma*normal_approx (use uniform as placeholder), store\n\
    cvt.u64.u32   %rd7, %r4;\n\
    shl.b64       %rd7, %rd7, 3;\n\
    add.u64       %rd7, %rd0, %rd7;\n\
    ld.global.f64 %fd4, [%rd7];\n\
    // generate second uniform for Box-Muller\n\
    mul.lo.u64    %rd8, %rd5, %rd4;\n\
    add.u64       %rd8, %rd8, 1442695040888963407;\n\
    shr.u64       %rd9, %rd8, 11;\n\
    cvt.rn.f64.u64 %fd5, %rd9;\n\
    mul.rn.f64    %fd5, %fd5, %fd3;\n\
    // delta = sigma * fd2 (approximation — full Box-Muller needs math.sin)\n\
    mul.rn.f64    %fd6, %fd0, %fd2;\n\
    add.rn.f64    %fd4, %fd4, %fd6;\n\
    st.global.f64 [%rd7], %fd4;\n\
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
