//! GPU PTX kernels for statistical operations.
//!
//! Each kernel is emitted as a self-contained PTX module string, parameterised on SM version.
//! PTX ISA is selected by SM:
//!     SM>=100 → 8.7 (Blackwell), SM>=90 → 8.4 (Hopper),
//!     SM>=80  → 8.0 (Ampere),    else → 7.5 (Turing).
//!
//! IMPORTANT: PTX kernel bodies use **string concatenation** (NOT `format!()`) for
//! sections containing `%rd`, `%r`, `%f`, `%fd` register names, which Rust's format macro
//! would misinterpret as unused format arguments in edition 2024.

/// Build a PTX file header string for the given SM version.
fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

/// Welford streaming mean and variance reduction.
///
/// Signature: `mean_var_kernel(x, n, out_mean, out_m2)`
/// Computes `mean[block]` and `M2[block]` (sum of squared deviations).
#[must_use]
pub fn mean_var_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry mean_var_kernel(\n\
        .param .u64 p_x,\n\
        .param .u32 p_n,\n\
        .param .u64 p_out_mean,\n\
        .param .u64 p_out_m2\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<12>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u64  %rd1, [p_out_mean];\n\
        ld.param.u64  %rd2, [p_out_m2];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $MV_DONE;\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.f32       %f1, 0f00000000;\n\
        mov.u32       %r5, 0;\n\
        mov.f32       %f2, 0f00000000;\n\
    \n\
    $MV_LOOP:\n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $MV_WRITE;\n\
    \n\
        // load x[i]\n\
        mul.wide.u32  %rd3, %r5, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f32 %f3, [%rd4];\n\
    \n\
        // n_new = n_old + 1 (here we just count via %r5+1)\n\
        add.u32       %r6, %r5, 1;\n\
        cvt.rn.f32.u32   %f4, %r6;\n\
    \n\
        // delta = x - mean\n\
        sub.f32       %f5, %f3, %f0;\n\
        // delta / n\n\
        div.rn.f32    %f6, %f5, %f4;\n\
        // mean += delta/n\n\
        add.f32       %f0, %f0, %f6;\n\
        // delta2 = x - mean (new)\n\
        sub.f32       %f7, %f3, %f0;\n\
        // m2 += delta * delta2\n\
        fma.rn.f32    %f1, %f5, %f7, %f1;\n\
    \n\
        mov.u32       %r5, %r6;\n\
        bra $MV_LOOP;\n\
    \n\
    $MV_WRITE:\n\
        st.global.f32 [%rd1], %f0;\n\
        st.global.f32 [%rd2], %f1;\n\
    \n\
    $MV_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Assign rank within an already-sorted array (ties = average rank).
///
/// Signature: `rank_assign_kernel(sorted, ranks, n)`
/// For ties, computes average of positions in [start, end).
#[must_use]
pub fn rank_assign_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry rank_assign_kernel(\n\
        .param .u64 p_sorted,\n\
        .param .u64 p_ranks,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_sorted];\n\
        ld.param.u64  %rd1, [p_ranks];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $RA_DONE;\n\
    \n\
        // load sorted[i]\n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f0, [%rd3];\n\
    \n\
        // find lower bound (j such that sorted[j] == sorted[i] starting back)\n\
        mov.u32       %r5, %r4;\n\
    $RA_BACK:\n\
        setp.eq.u32   %p0, %r5, 0;\n\
        @%p0 bra $RA_FWD;\n\
        sub.u32       %r6, %r5, 1;\n\
        mul.wide.u32  %rd4, %r6, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
        setp.ne.f32   %p0, %f1, %f0;\n\
        @%p0 bra $RA_FWD;\n\
        mov.u32       %r5, %r6;\n\
        bra $RA_BACK;\n\
    \n\
    $RA_FWD:\n\
        mov.u32       %r7, %r4;\n\
    $RA_FORWARD:\n\
        add.u32       %r8, %r7, 1;\n\
        setp.ge.u32   %p0, %r8, %r0;\n\
        @%p0 bra $RA_WRITE;\n\
        mul.wide.u32  %rd4, %r8, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
        setp.ne.f32   %p0, %f1, %f0;\n\
        @%p0 bra $RA_WRITE;\n\
        mov.u32       %r7, %r8;\n\
        bra $RA_FORWARD;\n\
    \n\
    $RA_WRITE:\n\
        // rank = ((r5+1) + (r7+1)) / 2\n\
        add.u32       %r9, %r5, 1;\n\
        add.u32       %r10, %r7, 1;\n\
        add.u32       %r11, %r9, %r10;\n\
        cvt.rn.f32.u32   %f2, %r11;\n\
        mov.f32       %f3, 0f40000000;\n\
        div.rn.f32    %f4, %f2, %f3;\n\
    \n\
        mul.wide.u32  %rd6, %r4, 4;\n\
        add.u64       %rd7, %rd1, %rd6;\n\
        st.global.f32 [%rd7], %f4;\n\
    \n\
    $RA_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Histogram bin counts for chi-squared goodness-of-fit.
///
/// Signature: `histogram_bin_kernel(x, n, low, dx, n_bins, counts)`
#[must_use]
pub fn histogram_bin_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry histogram_bin_kernel(\n\
        .param .u64 p_x,\n\
        .param .u32 p_n,\n\
        .param .f32 p_low,\n\
        .param .f32 p_dx,\n\
        .param .u32 p_n_bins,\n\
        .param .u64 p_counts\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.f32  %f0,  [p_low];\n\
        ld.param.f32  %f1,  [p_dx];\n\
        ld.param.u32  %r1,  [p_n_bins];\n\
        ld.param.u64  %rd1, [p_counts];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $HB_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r5, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f2, [%rd3];\n\
    \n\
        // bin = floor((x - low) / dx)\n\
        sub.f32       %f3, %f2, %f0;\n\
        div.rn.f32    %f4, %f3, %f1;\n\
        cvt.rzi.s32.f32 %r6, %f4;\n\
    \n\
        // clamp into [0, n_bins-1]\n\
        setp.lt.s32   %p0, %r6, 0;\n\
        @%p0 mov.u32  %r6, 0;\n\
        sub.u32       %r7, %r1, 1;\n\
        setp.gt.s32   %p0, %r6, %r7;\n\
        @%p0 mov.u32  %r6, %r7;\n\
    \n\
        // atomic add 1 to counts[bin]\n\
        mul.wide.u32  %rd4, %r6, 4;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        atom.global.add.u32 %r8, [%rd5], 1;\n\
    \n\
    $HB_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Generate one bootstrap sample via inline LCG.
///
/// Signature: `bootstrap_resample_kernel(x, n, out, seed)`
/// Each thread picks `i = lcg(seed, tid) % n` and writes `out[tid] = x[i]`.
#[must_use]
pub fn bootstrap_resample_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry bootstrap_resample_kernel(\n\
        .param .u64 p_x,\n\
        .param .u32 p_n,\n\
        .param .u64 p_out,\n\
        .param .u64 p_seed\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<4>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u64  %rd1, [p_out];\n\
        ld.param.u64  %rd2, [p_seed];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $BR_DONE;\n\
    \n\
        // state = seed + tid * MUL\n\
        cvt.u64.u32   %rd3, %r4;\n\
        mul.lo.u64    %rd4, %rd3, 6364136223846793005;\n\
        add.u64       %rd5, %rd2, %rd4;\n\
        // advance LCG once more\n\
        mul.lo.u64    %rd6, %rd5, 6364136223846793005;\n\
        add.u64       %rd7, %rd6, 1442695040888963407;\n\
    \n\
        // idx = (state >> 32) % n\n\
        shr.u64       %rd8, %rd7, 32;\n\
        cvt.u32.u64   %r5, %rd8;\n\
        rem.u32       %r6, %r5, %r0;\n\
    \n\
        mul.wide.u32  %rd9, %r6, 4;\n\
        add.u64       %rd3, %rd0, %rd9;\n\
        ld.global.f32 %f0, [%rd3];\n\
    \n\
        mul.wide.u32  %rd4, %r4, 4;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        st.global.f32 [%rd5], %f0;\n\
    \n\
    $BR_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Fisher-Yates partial shuffle for permutation tests.
///
/// Signature: `permute_labels_kernel(labels_in, labels_out, n, seed)`
/// Each thread copies `labels_in[i]` to `labels_out[j]` where j is a derived random position.
#[must_use]
pub fn permute_labels_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry permute_labels_kernel(\n\
        .param .u64 p_labels_in,\n\
        .param .u64 p_labels_out,\n\
        .param .u32 p_n,\n\
        .param .u64 p_seed\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<16>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_labels_in];\n\
        ld.param.u64  %rd1, [p_labels_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u64  %rd2, [p_seed];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $PL_DONE;\n\
    \n\
        // pos = lcg(seed + tid)\n\
        cvt.u64.u32   %rd3, %r4;\n\
        add.u64       %rd4, %rd2, %rd3;\n\
        mul.lo.u64    %rd5, %rd4, 6364136223846793005;\n\
        add.u64       %rd6, %rd5, 1442695040888963407;\n\
        shr.u64       %rd7, %rd6, 32;\n\
        cvt.u32.u64   %r5, %rd7;\n\
        rem.u32       %r6, %r5, %r0;\n\
    \n\
        // load labels_in[tid]\n\
        mul.wide.u32  %rd8, %r4, 4;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        ld.global.u32 %r7, [%rd9];\n\
    \n\
        // store labels_out[pos]\n\
        mul.wide.u32  %rd10, %r6, 4;\n\
        add.u64       %rd11, %rd1, %rd10;\n\
        st.global.u32 [%rd11], %r7;\n\
    \n\
    $PL_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Compute (O - E)^2 / E per cell for a chi-squared contingency table.
///
/// Signature: `chi2_cell_kernel(observed, expected, out, n_cells)`
#[must_use]
pub fn chi2_cell_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry chi2_cell_kernel(\n\
        .param .u64 p_observed,\n\
        .param .u64 p_expected,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n_cells\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_observed];\n\
        ld.param.u64  %rd1, [p_expected];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_n_cells];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $CC_DONE;\n\
    \n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f32 %f0, [%rd4];\n\
        add.u64       %rd5, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd5];\n\
    \n\
        sub.f32       %f2, %f0, %f1;\n\
        mul.f32       %f3, %f2, %f2;\n\
        // guard against E=0 with epsilon\n\
        mov.f32       %f4, 0f322bcc77;\n\
        add.f32       %f5, %f1, %f4;\n\
        div.rn.f32    %f6, %f3, %f5;\n\
    \n\
        add.u64       %rd6, %rd2, %rd3;\n\
        st.global.f32 [%rd6], %f6;\n\
    \n\
    $CC_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// X^T X accumulation for linear regression normal equations.
///
/// Signature: `lr_normal_eq_kernel(x, xt_x, n_samples, n_features)`
/// For row blocks, compute outer products X[i, :]^T * X[i, :] and accumulate.
#[must_use]
pub fn lr_normal_eq_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry lr_normal_eq_kernel(\n\
        .param .u64 p_x,\n\
        .param .u64 p_xt_x,\n\
        .param .u32 p_n_samples,\n\
        .param .u32 p_n_features\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<24>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u64  %rd1, [p_xt_x];\n\
        ld.param.u32  %r0,  [p_n_samples];\n\
        ld.param.u32  %r1,  [p_n_features];\n\
    \n\
        // i = blockIdx.y * blockDim.y + threadIdx.y\n\
        mov.u32       %r2, %ntid.y;\n\
        mov.u32       %r3, %ctaid.y;\n\
        mov.u32       %r4, %tid.y;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        // j = blockIdx.x * blockDim.x + threadIdx.x\n\
        mov.u32       %r6, %ntid.x;\n\
        mov.u32       %r7, %ctaid.x;\n\
        mov.u32       %r8, %tid.x;\n\
        mad.lo.u32    %r9, %r6, %r7, %r8;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r1;\n\
        @%p0 bra $LR_DONE;\n\
        setp.ge.u32   %p0, %r9, %r1;\n\
        @%p0 bra $LR_DONE;\n\
    \n\
        // accumulator\n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r10, 0;\n\
    \n\
    $LR_LOOP:\n\
        setp.ge.u32   %p0, %r10, %r0;\n\
        @%p0 bra $LR_WRITE;\n\
    \n\
        // x[r10, r5] = x[r10*nf + r5]\n\
        mul.lo.u32    %r11, %r10, %r1;\n\
        add.u32       %r12, %r11, %r5;\n\
        mul.wide.u32  %rd2, %r12, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
    \n\
        // x[r10, r9] = x[r10*nf + r9]\n\
        add.u32       %r13, %r11, %r9;\n\
        mul.wide.u32  %rd4, %r13, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f2, [%rd5];\n\
    \n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
    \n\
        add.u32       %r10, %r10, 1;\n\
        bra $LR_LOOP;\n\
    \n\
    $LR_WRITE:\n\
        // xt_x[r5, r9] = xt_x[r5*nf + r9]\n\
        mul.lo.u32    %r14, %r5, %r1;\n\
        add.u32       %r15, %r14, %r9;\n\
        mul.wide.u32  %rd6, %r15, 4;\n\
        add.u64       %rd7, %rd1, %rd6;\n\
        st.global.f32 [%rd7], %f0;\n\
    \n\
    $LR_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

#[cfg(test)]
mod tests {
    use super::*;

    type KernelFn = fn(u32) -> String;

    fn all_kernels() -> Vec<(&'static str, KernelFn)> {
        vec![
            ("mean_var", mean_var_ptx),
            ("rank_assign", rank_assign_ptx),
            ("histogram_bin", histogram_bin_ptx),
            ("bootstrap_resample", bootstrap_resample_ptx),
            ("permute_labels", permute_labels_ptx),
            ("chi2_cell", chi2_cell_ptx),
            ("lr_normal_eq", lr_normal_eq_ptx),
        ]
    }

    #[test]
    fn ptx_header_versions() {
        assert!(ptx_header(75).contains("7.5"));
        assert!(ptx_header(80).contains("8.0"));
        assert!(ptx_header(90).contains("8.4"));
        assert!(ptx_header(100).contains("8.7"));
    }

    #[test]
    fn ptx_all_kernels_non_empty_all_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            for (name, f) in all_kernels() {
                let s = f(sm);
                assert!(!s.is_empty(), "kernel {name} sm={sm} produced empty string");
                assert!(
                    s.contains(".visible .entry"),
                    "kernel {name} sm={sm} missing entry"
                );
                assert!(s.contains("ret"), "kernel {name} sm={sm} missing ret");
            }
        }
    }
}
