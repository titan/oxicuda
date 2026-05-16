//! GPU PTX kernels for streaming sketch operations.
//!
//! Each kernel is emitted as a self-contained PTX module string, parameterised on SM version.
//! PTX ISA is selected by SM:
//!     SM>=100 -> 8.7 (Blackwell), SM>=90 -> 8.4 (Hopper),
//!     SM>=80  -> 8.0 (Ampere),    else   -> 7.5 (Turing).
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

/// Count-Min Sketch insert kernel: each thread updates one row of the table.
///
/// Signature: `cm_update_kernel(table, hashes_ai, hashes_bi, w, d, x)`
/// table[row, h_row(x) mod w] += 1, where h_row(x) = (a_row * x + b_row) mod prime mod w
#[must_use]
pub fn cm_update_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry cm_update_kernel(\n\
        .param .u64 p_table,\n\
        .param .u64 p_a,\n\
        .param .u64 p_b,\n\
        .param .u32 p_w,\n\
        .param .u32 p_d,\n\
        .param .u64 p_x\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<16>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_table];\n\
        ld.param.u64  %rd1, [p_a];\n\
        ld.param.u64  %rd2, [p_b];\n\
        ld.param.u32  %r0,  [p_w];\n\
        ld.param.u32  %r1,  [p_d];\n\
        ld.param.u64  %rd3, [p_x];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r1;\n\
        @%p0 bra $CM_UPD_DONE;\n\
    \n\
        // load a[row], b[row]\n\
        mul.wide.u32  %rd4, %r5, 8;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        ld.global.u64 %rd6, [%rd5];\n\
        add.u64       %rd7, %rd2, %rd4;\n\
        ld.global.u64 %rd8, [%rd7];\n\
    \n\
        // h = (a * x + b) mod 2^61-1, then mod w\n\
        mul.lo.u64    %rd9, %rd6, %rd3;\n\
        add.u64       %rd9, %rd9, %rd8;\n\
        // approximate modulo via low 32 bits then mod w\n\
        cvt.u32.u64   %r6, %rd9;\n\
        rem.u32       %r7, %r6, %r0;\n\
    \n\
        // table[row * w + col] += 1\n\
        mul.lo.u32    %r8, %r5, %r0;\n\
        add.u32       %r8, %r8, %r7;\n\
        mul.wide.u32  %rd10, %r8, 4;\n\
        add.u64       %rd11, %rd0, %rd10;\n\
        atom.global.add.u32 %r9, [%rd11], 1;\n\
    \n\
    $CM_UPD_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Count-Min Sketch query kernel: read d row entries and reduce-min to global output.
///
/// Signature: `cm_query_kernel(table, hashes_a, hashes_b, w, d, x, out_min)`
#[must_use]
pub fn cm_query_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry cm_query_kernel(\n\
        .param .u64 p_table,\n\
        .param .u64 p_a,\n\
        .param .u64 p_b,\n\
        .param .u32 p_w,\n\
        .param .u32 p_d,\n\
        .param .u64 p_x,\n\
        .param .u64 p_out\n\
    )\n\
    {\n\
        .reg .u64  %rd<14>;\n\
        .reg .u32  %r<20>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_table];\n\
        ld.param.u64  %rd1, [p_a];\n\
        ld.param.u64  %rd2, [p_b];\n\
        ld.param.u32  %r0,  [p_w];\n\
        ld.param.u32  %r1,  [p_d];\n\
        ld.param.u64  %rd3, [p_x];\n\
        ld.param.u64  %rd12, [p_out];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r1;\n\
        @%p0 bra $CM_Q_DONE;\n\
    \n\
        mul.wide.u32  %rd4, %r5, 8;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        ld.global.u64 %rd6, [%rd5];\n\
        add.u64       %rd7, %rd2, %rd4;\n\
        ld.global.u64 %rd8, [%rd7];\n\
    \n\
        mul.lo.u64    %rd9, %rd6, %rd3;\n\
        add.u64       %rd9, %rd9, %rd8;\n\
        cvt.u32.u64   %r6, %rd9;\n\
        rem.u32       %r7, %r6, %r0;\n\
    \n\
        mul.lo.u32    %r8, %r5, %r0;\n\
        add.u32       %r8, %r8, %r7;\n\
        mul.wide.u32  %rd10, %r8, 4;\n\
        add.u64       %rd11, %rd0, %rd10;\n\
        ld.global.u32 %r9, [%rd11];\n\
    \n\
        // atomic-min into out\n\
        atom.global.min.u32 %r10, [%rd12], %r9;\n\
    \n\
    $CM_Q_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// HyperLogLog register update kernel: compute leading-zero count and atomic-max into register.
///
/// Signature: `hll_register_kernel(registers, m, p_bits, hash_x)`
/// `idx` = top p_bits of hash_x, `register[idx]` = max(`register[idx]`, lz(low (64 - p_bits) bits) + 1).
#[must_use]
pub fn hll_register_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry hll_register_kernel(\n\
        .param .u64 p_regs,\n\
        .param .u32 p_m,\n\
        .param .u32 p_p_bits,\n\
        .param .u64 p_hash\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<24>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_regs];\n\
        ld.param.u32  %r0,  [p_m];\n\
        ld.param.u32  %r1,  [p_p_bits];\n\
        ld.param.u64  %rd1, [p_hash];\n\
    \n\
        // only one thread does the update\n\
        mov.u32       %r2, %tid.x;\n\
        setp.ne.u32   %p0, %r2, 0;\n\
        @%p0 bra $HLL_DONE;\n\
    \n\
        // idx = hash >> (64 - p_bits)  (use low 32 bits trick: idx = (hash * m) >> 32 for m=2^p)\n\
        // For simplicity assume m = 2^p_bits; idx = (hash >> 32) & (m-1)\n\
        cvt.u32.u64   %r3, %rd1;\n\
        sub.u32       %r4, %r0, 1;\n\
        and.b32       %r5, %r3, %r4;\n\
    \n\
        // hash_lo = hash >> p_bits; lz = clz(hash_lo) + 1 (capped to 64-p_bits)\n\
        shr.u64       %rd2, %rd1, %r1;\n\
        cvt.u32.u64   %r6, %rd2;\n\
        clz.b32       %r7, %r6;\n\
        add.u32       %r8, %r7, 1;\n\
    \n\
        // atomic-max into register[idx] as 4-byte u32\n\
        mul.wide.u32  %rd3, %r5, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        atom.global.max.u32 %r9, [%rd4], %r8;\n\
    \n\
    $HLL_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Bloom filter insert kernel: for each of k hashes set the corresponding bit.
///
/// Signature: `bloom_insert_kernel(bits_u32, m_bits, k, seeds, x)`
#[must_use]
pub fn bloom_insert_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry bloom_insert_kernel(\n\
        .param .u64 p_bits,\n\
        .param .u32 p_m,\n\
        .param .u32 p_k,\n\
        .param .u64 p_seeds,\n\
        .param .u64 p_x\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<24>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_bits];\n\
        ld.param.u32  %r0,  [p_m];\n\
        ld.param.u32  %r1,  [p_k];\n\
        ld.param.u64  %rd1, [p_seeds];\n\
        ld.param.u64  %rd2, [p_x];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r1;\n\
        @%p0 bra $BL_DONE;\n\
    \n\
        // load seed[i]\n\
        mul.wide.u32  %rd3, %r5, 8;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.u64 %rd5, [%rd4];\n\
    \n\
        // h = seed * x  (multiplicative hash low 32)\n\
        mul.lo.u64    %rd6, %rd5, %rd2;\n\
        cvt.u32.u64   %r6, %rd6;\n\
        rem.u32       %r7, %r6, %r0;\n\
    \n\
        // word_idx = bit / 32, bit_off = bit % 32\n\
        shr.u32       %r8, %r7, 5;\n\
        and.b32       %r9, %r7, 31;\n\
        mov.u32       %r10, 1;\n\
        shl.b32       %r11, %r10, %r9;\n\
    \n\
        // atomic-or into bits[word_idx]\n\
        mul.wide.u32  %rd7, %r8, 4;\n\
        add.u64       %rd8, %rd0, %rd7;\n\
        atom.global.or.b32 %r12, [%rd8], %r11;\n\
    \n\
    $BL_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// MinHash signature kernel: compute `signature[i]` = min(`signature[i]`, h_i(x)) for K hash functions.
///
/// Signature: `minhash_sketch_kernel(signature, seeds_a, seeds_b, k_hashes, x)`
#[must_use]
pub fn minhash_sketch_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry minhash_sketch_kernel(\n\
        .param .u64 p_sig,\n\
        .param .u64 p_a,\n\
        .param .u64 p_b,\n\
        .param .u32 p_k,\n\
        .param .u64 p_x\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<16>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_sig];\n\
        ld.param.u64  %rd1, [p_a];\n\
        ld.param.u64  %rd2, [p_b];\n\
        ld.param.u32  %r0,  [p_k];\n\
        ld.param.u64  %rd3, [p_x];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $MH_DONE;\n\
    \n\
        mul.wide.u32  %rd4, %r4, 8;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        ld.global.u64 %rd6, [%rd5];\n\
        add.u64       %rd7, %rd2, %rd4;\n\
        ld.global.u64 %rd8, [%rd7];\n\
    \n\
        mul.lo.u64    %rd9, %rd6, %rd3;\n\
        add.u64       %rd9, %rd9, %rd8;\n\
        cvt.u32.u64   %r5, %rd9;\n\
    \n\
        // signature[i] = min(signature[i], h)\n\
        mul.wide.u32  %rd10, %r4, 4;\n\
        add.u64       %rd11, %rd0, %rd10;\n\
        atom.global.min.u32 %r6, [%rd11], %r5;\n\
    \n\
    $MH_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// t-Digest centroid merge step kernel: merge centroid `i` into centroid `j` updating mean and weight.
///
/// Signature: `tdigest_merge_kernel(means, weights, idx_i, idx_j)`
/// new_w = w_i + w_j; new_m = (w_i*m_i + w_j*m_j) / new_w. Then `weight[i]` = 0.
#[must_use]
pub fn tdigest_merge_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry tdigest_merge_kernel(\n\
        .param .u64 p_means,\n\
        .param .u64 p_weights,\n\
        .param .u32 p_i,\n\
        .param .u32 p_j\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<8>;\n\
        .reg .f64  %fd<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_means];\n\
        ld.param.u64  %rd1, [p_weights];\n\
        ld.param.u32  %r0,  [p_i];\n\
        ld.param.u32  %r1,  [p_j];\n\
    \n\
        mov.u32       %r2, %tid.x;\n\
        setp.ne.u32   %p0, %r2, 0;\n\
        @%p0 bra $TD_DONE;\n\
    \n\
        // load means/weights at i and j\n\
        mul.wide.u32  %rd2, %r0, 8;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f64 %fd0, [%rd3];\n\
        add.u64       %rd4, %rd1, %rd2;\n\
        ld.global.f64 %fd1, [%rd4];\n\
    \n\
        mul.wide.u32  %rd5, %r1, 8;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f64 %fd2, [%rd6];\n\
        add.u64       %rd7, %rd1, %rd5;\n\
        ld.global.f64 %fd3, [%rd7];\n\
    \n\
        // new_w = w_i + w_j; new_m = (w_i*m_i + w_j*m_j) / new_w\n\
        add.f64       %fd4, %fd1, %fd3;\n\
        mul.f64       %fd5, %fd0, %fd1;\n\
        fma.rn.f64    %fd6, %fd2, %fd3, %fd5;\n\
        div.rn.f64    %fd7, %fd6, %fd4;\n\
    \n\
        // store merged values into j, zero into i\n\
        st.global.f64 [%rd6], %fd7;\n\
        st.global.f64 [%rd7], %fd4;\n\
    \n\
        mov.f64       %fd8, 0d0000000000000000;\n\
        st.global.f64 [%rd3], %fd8;\n\
        st.global.f64 [%rd4], %fd8;\n\
    \n\
    $TD_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Reservoir-sampling replacement step kernel: with prob k/i replace `reservoir[j]` with item.
///
/// Signature: `reservoir_sample_kernel(reservoir, k, index_i, random_u, item)`
/// `j` = random_u % i; if j < k, `reservoir[j]` = item.
#[must_use]
pub fn reservoir_sample_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry reservoir_sample_kernel(\n\
        .param .u64 p_res,\n\
        .param .u32 p_k,\n\
        .param .u32 p_i,\n\
        .param .u32 p_rand,\n\
        .param .u64 p_item\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_res];\n\
        ld.param.u32  %r0,  [p_k];\n\
        ld.param.u32  %r1,  [p_i];\n\
        ld.param.u32  %r2,  [p_rand];\n\
        ld.param.u64  %rd1, [p_item];\n\
    \n\
        mov.u32       %r3, %tid.x;\n\
        setp.ne.u32   %p0, %r3, 0;\n\
        @%p0 bra $RS_DONE;\n\
    \n\
        // j = rand % i (i >= 1)\n\
        rem.u32       %r4, %r2, %r1;\n\
    \n\
        // if j < k: reservoir[j] = item\n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $RS_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 8;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        st.global.u64 [%rd3], %rd1;\n\
    \n\
    $RS_DONE:\n\
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
            ("cm_update", cm_update_ptx),
            ("cm_query", cm_query_ptx),
            ("hll_register", hll_register_ptx),
            ("bloom_insert", bloom_insert_ptx),
            ("minhash_sketch", minhash_sketch_ptx),
            ("tdigest_merge", tdigest_merge_ptx),
            ("reservoir_sample", reservoir_sample_ptx),
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
