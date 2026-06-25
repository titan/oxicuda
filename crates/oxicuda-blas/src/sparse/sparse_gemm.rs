//! Sparse GEMM (SpGEMM) via 2:4 structured sparsity.
//!
//! Implements the Ampere+ (sm_80) *structured sparse* matrix multiply used by
//! the sparse Tensor Cores. The format compresses the operand `A` by exploiting
//! a **2:4 sparsity pattern**: within every contiguous group of four elements
//! along the contraction (`K`) dimension **at most two are non-zero**. This
//! halves both the storage and the number of multiply–accumulate operations,
//! while a compact 2-bit-per-kept-element *metadata* index records where each
//! surviving value lives inside its group of four.
//!
//! ```text
//!   dense group of 4 :  [ a0  0  a2  0 ]      (2 non-zeros out of 4)
//!   compressed values:  [ a0 a2 ]            (densely packed)
//!   metadata indices :  [  0  2 ]            (2-bit lane index, 0..=3)
//! ```
//!
//! On hardware the kernel issues `mma.sp.sync.aligned …` (sparse `mma`), feeding
//! the compressed `A` fragment plus its metadata so the Tensor Core skips the
//! zero lanes. This module provides:
//!
//! * a **pure-CPU reference** ([`compress_2to4`], [`decompress_2to4`],
//!   [`spgemm_2to4`]) that is numerically validated against a dense GEMM, and
//! * a **PTX-string generator** ([`generate_sparse_gemm_ptx`]) emitting the
//!   `mma.sp`-based sparse Tensor-Core kernel source (content-checkable on the
//!   host; the device launch path lives behind the GPU-gated kernels).
//!
//! # References
//! - NVIDIA, *Accelerating Inference with Sparsity Using the NVIDIA Ampere
//!   Architecture and NVIDIA TensorRT* (2:4 structured sparsity).
//! - NVIDIA PTX ISA, `mma.sp` sparse matrix multiply-accumulate.

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::prelude::SmVersion;

use crate::error::{BlasError, BlasResult};

// ===========================================================================
// 2:4 metadata
// ===========================================================================

/// Number of elements in a single 2:4 sparsity group.
pub const GROUP: usize = 4;

/// Maximum number of non-zeros kept per 2:4 group.
pub const KEPT: usize = 2;

/// The 2-bit lane indices of the (at most two) kept elements within one group
/// of four, in ascending order.
///
/// A value of `[i, j]` means the compressed pair `[v0, v1]` corresponds to
/// dense lanes `i` and `j` (`0 <= i < j <= 3`). When a group has fewer than two
/// non-zeros, the trailing slot stores a duplicate of the last real index and
/// its compressed value is `0`, mirroring the hardware convention that padding
/// lanes contribute nothing to the dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TwoFourMeta {
    /// Ascending 2-bit lane indices (each `0..=3`) of the kept elements.
    pub lanes: [u8; KEPT],
}

impl TwoFourMeta {
    /// Packs the two 2-bit lane indices into the low nibble of a `u8`
    /// (`lanes[0]` in bits `0..2`, `lanes[1]` in bits `2..4`), matching the
    /// per-group layout the sparse `mma` metadata operand expects.
    #[must_use]
    pub const fn pack_nibble(self) -> u8 {
        (self.lanes[0] & 0b11) | ((self.lanes[1] & 0b11) << 2)
    }

    /// Reconstructs a [`TwoFourMeta`] from a packed low nibble.
    #[must_use]
    pub const fn from_nibble(nibble: u8) -> Self {
        Self {
            lanes: [nibble & 0b11, (nibble >> 2) & 0b11],
        }
    }
}

/// Packs a slice of per-group [`TwoFourMeta`] into the dense `u32` metadata
/// words consumed by `mma.sp`.
///
/// Eight groups (sixteen 2-bit indices) are packed per `u32`, low group first:
/// `[g0:4][g1:4]…[g7:4]`.
///
/// # Errors
/// Never fails; returns the packed words. Provided as `BlasResult` for API
/// symmetry with the rest of the crate.
pub fn pack_metadata(meta: &[TwoFourMeta]) -> Vec<u32> {
    let words = meta.len().div_ceil(8);
    let mut out = vec![0u32; words];
    for (g, m) in meta.iter().enumerate() {
        let word = g / 8;
        let shift = (g % 8) * 4;
        out[word] |= u32::from(m.pack_nibble()) << shift;
    }
    out
}

/// Unpacks `count` per-group [`TwoFourMeta`] from packed `u32` metadata words.
#[must_use]
pub fn unpack_metadata(words: &[u32], count: usize) -> Vec<TwoFourMeta> {
    (0..count)
        .map(|g| {
            let word = g / 8;
            let shift = (g % 8) * 4;
            let nibble = ((words[word] >> shift) & 0b1111) as u8;
            TwoFourMeta::from_nibble(nibble)
        })
        .collect()
}

// ===========================================================================
// Compression / decompression
// ===========================================================================

/// A matrix compressed to the 2:4 structured-sparse format along its columns.
///
/// Stored row-major. For a logical `m x k` matrix with `k` a multiple of
/// [`GROUP`], the compressed value array has `m * (k / 2)` entries (two kept
/// values per group of four), and the metadata has `m * (k / 4)` entries (one
/// [`TwoFourMeta`] per group).
#[derive(Debug, Clone, PartialEq)]
pub struct Compressed2to4 {
    /// Logical row count.
    pub rows: usize,
    /// Logical (dense) column count; multiple of [`GROUP`].
    pub cols: usize,
    /// Densely-packed kept values, row-major, length `rows * (cols / 2)`.
    pub values: Vec<f32>,
    /// Per-group lane metadata, row-major, length `rows * (cols / 4)`.
    pub meta: Vec<TwoFourMeta>,
}

impl Compressed2to4 {
    /// Number of 2:4 groups per row (`cols / 4`).
    #[must_use]
    pub const fn groups_per_row(&self) -> usize {
        self.cols / GROUP
    }

    /// Number of compressed values per row (`cols / 2`).
    #[must_use]
    pub const fn values_per_row(&self) -> usize {
        self.cols / GROUP * KEPT
    }
}

/// Compresses a dense `rows x cols` row-major matrix to the 2:4 structured
/// format, keeping the two largest-magnitude elements of each group of four.
///
/// This is the standard *magnitude pruning* used to coerce an arbitrary dense
/// matrix into a 2:4 pattern: within each group of four consecutive column
/// elements the two with the smallest absolute value are dropped (set to zero)
/// and the survivors are packed densely with their lane indices recorded.
///
/// Ties are broken toward the lower lane index, giving a deterministic result.
///
/// # Errors
/// * [`BlasError::InvalidDimension`] if `rows == 0` or `cols == 0`.
/// * [`BlasError::InvalidArgument`] if `cols` is not a multiple of [`GROUP`].
/// * [`BlasError::DimensionMismatch`] if `dense.len() != rows * cols`.
pub fn compress_2to4(dense: &[f32], rows: usize, cols: usize) -> BlasResult<Compressed2to4> {
    if rows == 0 || cols == 0 {
        return Err(BlasError::InvalidDimension(format!(
            "compress_2to4: rows and cols must be >= 1 (got rows={rows}, cols={cols})"
        )));
    }
    if cols % GROUP != 0 {
        return Err(BlasError::InvalidArgument(format!(
            "compress_2to4: cols must be a multiple of {GROUP} (got {cols})"
        )));
    }
    if dense.len() != rows * cols {
        return Err(BlasError::DimensionMismatch(format!(
            "compress_2to4: dense has {} elements, expected rows*cols = {}",
            dense.len(),
            rows * cols
        )));
    }

    let groups = cols / GROUP;
    let mut values = Vec::with_capacity(rows * groups * KEPT);
    let mut meta = Vec::with_capacity(rows * groups);

    for r in 0..rows {
        for g in 0..groups {
            let base = r * cols + g * GROUP;
            let group = [
                dense[base],
                dense[base + 1],
                dense[base + 2],
                dense[base + 3],
            ];
            // Select the two lanes with the largest magnitude. Ties break to
            // the lower lane index so the result is deterministic.
            let mut idx = [0usize, 1, 2, 3];
            idx.sort_by(|&i, &j| {
                let (mi, mj) = (group[i].abs(), group[j].abs());
                // Descending magnitude; lower lane index wins ties.
                mj.partial_cmp(&mi)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(i.cmp(&j))
            });
            let mut kept = [idx[0], idx[1]];
            kept.sort_unstable(); // ascending lane order, as hardware expects
            values.push(group[kept[0]]);
            values.push(group[kept[1]]);
            meta.push(TwoFourMeta {
                lanes: [kept[0] as u8, kept[1] as u8],
            });
        }
    }

    Ok(Compressed2to4 {
        rows,
        cols,
        values,
        meta,
    })
}

/// Reconstructs the dense `rows x cols` row-major matrix from its 2:4
/// compressed form, scattering each kept value back to its recorded lane and
/// zero-filling the dropped lanes.
///
/// `decompress_2to4(compress_2to4(dense))` recovers `dense` exactly whenever
/// `dense` already obeys a 2:4 pattern; otherwise it recovers the pruned
/// matrix (dropped lanes become `0`).
///
/// # Errors
/// * [`BlasError::DimensionMismatch`] if the value/metadata lengths are
///   inconsistent with the stored dimensions.
pub fn decompress_2to4(c: &Compressed2to4) -> BlasResult<Vec<f32>> {
    let groups = c.groups_per_row();
    if c.values.len() != c.rows * groups * KEPT {
        return Err(BlasError::DimensionMismatch(format!(
            "decompress_2to4: values has {} entries, expected {}",
            c.values.len(),
            c.rows * groups * KEPT
        )));
    }
    if c.meta.len() != c.rows * groups {
        return Err(BlasError::DimensionMismatch(format!(
            "decompress_2to4: meta has {} entries, expected {}",
            c.meta.len(),
            c.rows * groups
        )));
    }

    let mut dense = vec![0.0_f32; c.rows * c.cols];
    for r in 0..c.rows {
        for g in 0..groups {
            let gi = r * groups + g;
            let m = c.meta[gi];
            let vbase = gi * KEPT;
            let base = r * c.cols + g * GROUP;
            dense[base + m.lanes[0] as usize] = c.values[vbase];
            dense[base + m.lanes[1] as usize] = c.values[vbase + 1];
        }
    }
    Ok(dense)
}

// ===========================================================================
// Sparse GEMM reference
// ===========================================================================

/// Sparse GEMM `C = alpha * A_sparse * B + beta * C` (CPU reference).
///
/// * `a` — `m x k` operand in 2:4 compressed form (rows `m`, cols `k`).
/// * `b` — dense `k x n` row-major matrix, `k * n` elements.
/// * `c` — dense `m x n` row-major matrix (in/out), `m * n` elements.
///
/// The contraction iterates only over the kept (non-zero) lanes of each group
/// of four — exactly the work the sparse Tensor Core performs — so the result
/// matches a dense GEMM of the decompressed `A` against `B` up to floating
/// point. When `beta == 0` the prior contents of `C` are not read (so `NaN`/
/// uninitialised values are cleared).
///
/// # Errors
/// * [`BlasError::InvalidDimension`] if any dimension is zero.
/// * [`BlasError::DimensionMismatch`] if `b.len() != k*n` or `c.len() != m*n`,
///   or the compressed operand is internally inconsistent.
pub fn spgemm_2to4(
    a: &Compressed2to4,
    b: &[f32],
    n: usize,
    alpha: f32,
    beta: f32,
    c: &mut [f32],
) -> BlasResult<()> {
    let m = a.rows;
    let k = a.cols;
    if m == 0 || k == 0 || n == 0 {
        return Err(BlasError::InvalidDimension(format!(
            "spgemm_2to4: m, k, n must be >= 1 (got m={m}, k={k}, n={n})"
        )));
    }
    let groups = a.groups_per_row();
    if a.values.len() != m * groups * KEPT || a.meta.len() != m * groups {
        return Err(BlasError::DimensionMismatch(
            "spgemm_2to4: compressed A is internally inconsistent".to_string(),
        ));
    }
    if b.len() != k * n {
        return Err(BlasError::DimensionMismatch(format!(
            "spgemm_2to4: B has {} elements, expected k*n = {}",
            b.len(),
            k * n
        )));
    }
    if c.len() != m * n {
        return Err(BlasError::DimensionMismatch(format!(
            "spgemm_2to4: C has {} elements, expected m*n = {}",
            c.len(),
            m * n
        )));
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for g in 0..groups {
                let gi = i * groups + g;
                let meta = a.meta[gi];
                let vbase = gi * KEPT;
                // Two kept lanes per group; each maps to a row of B.
                let col0 = g * GROUP + meta.lanes[0] as usize;
                let col1 = g * GROUP + meta.lanes[1] as usize;
                acc += a.values[vbase] * b[col0 * n + j];
                acc += a.values[vbase + 1] * b[col1 * n + j];
            }
            let dst = &mut c[i * n + j];
            *dst = if beta == 0.0 {
                alpha * acc
            } else {
                alpha * acc + beta * *dst
            };
        }
    }
    Ok(())
}

// ===========================================================================
// Sparse GEMM configuration + PTX generation
// ===========================================================================

/// Configuration for a 2:4 structured-sparse Tensor-Core GEMM kernel.
///
/// Drives [`generate_sparse_gemm_ptx`]. The compressed operand `A` is `m x k`
/// (logical/dense `k`), `B` is dense `k x n`, and the output `C` is `m x n`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SparseGemmConfig {
    /// Rows of `A` / `C`.
    pub m: u32,
    /// Columns of `B` / `C`.
    pub n: u32,
    /// Dense contraction dimension (columns of `A`, rows of `B`).
    /// Must be a multiple of 4 (the 2:4 group size).
    pub k: u32,
}

impl SparseGemmConfig {
    /// Sparse `mma` is available on Ampere (sm_80) and newer.
    #[must_use]
    pub fn is_available(sm: SmVersion) -> bool {
        sm >= SmVersion::Sm80
    }

    /// Validates dimensions and architecture support.
    ///
    /// # Errors
    /// * [`BlasError::InvalidDimension`] if any dimension is zero.
    /// * [`BlasError::InvalidArgument`] if `k` is not a multiple of 4.
    /// * [`BlasError::UnsupportedOperation`] if `sm < sm_80`.
    pub fn validate(&self, sm: SmVersion) -> BlasResult<()> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(BlasError::InvalidDimension(format!(
                "sparse_gemm: m, n, k must be >= 1 (got m={}, n={}, k={})",
                self.m, self.n, self.k
            )));
        }
        if self.k % 4 != 0 {
            return Err(BlasError::InvalidArgument(format!(
                "sparse_gemm: k must be a multiple of 4 (got {})",
                self.k
            )));
        }
        if !Self::is_available(sm) {
            return Err(BlasError::UnsupportedOperation(format!(
                "sparse_gemm: 2:4 structured sparsity requires sm_80+ (got {})",
                sm.as_ptx_str()
            )));
        }
        Ok(())
    }

    /// Mangled kernel name encoding the problem shape.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!("spgemm_2to4_{}x{}x{}", self.m, self.n, self.k)
    }

    /// Compression ratio of the `A` operand (always `0.5` for 2:4).
    #[must_use]
    pub const fn compression_ratio() -> f32 {
        KEPT as f32 / GROUP as f32
    }
}

/// Generates PTX source for a 2:4 structured-sparse GEMM kernel.
///
/// The emitted kernel multiplies a 2:4-compressed `A` (values + 2-bit lane
/// metadata) by a dense `B` using the sparse Tensor-Core instruction
/// `mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (16x8x16 F16 inputs,
/// F32 accumulate) — the canonical Ampere sparse `mma` shape. Each warp owns a
/// 16x8 output tile; the `mma.sp` metadata operand selects the live lanes of
/// the compressed `A` fragment so the zero lanes are skipped.
///
/// This is a structurally-faithful reference: a production kernel adds shared
/// memory staging and `cp.async` pipelining. The generated string is validated
/// on the host (content checks); device execution lives behind the GPU-gated
/// launch path.
///
/// # Errors
/// Returns [`BlasError`] if validation fails or PTX formatting fails.
pub fn generate_sparse_gemm_ptx(config: &SparseGemmConfig, sm: SmVersion) -> BlasResult<String> {
    config.validate(sm)?;

    let kernel_name = config.kernel_name();
    // 16x8x16 sparse mma tile; K iterates in steps of 16 dense (8 compressed).
    let k_iters = config.k / 16;

    let mut ptx = String::with_capacity(8192);

    writeln(&mut ptx, &format!(".version {}", sm.ptx_version()))?;
    writeln(&mut ptx, &format!(".target {}", sm.as_ptx_str()))?;
    writeln(&mut ptx, ".address_size 64")?;
    writeln(&mut ptx, "")?;

    // Kernel signature: compressed A values, A metadata, dense B, C, dims.
    writeln(&mut ptx, &format!(".visible .entry {kernel_name}("))?;
    writeln(&mut ptx, "    .param .u64 %param_a_values,")?;
    writeln(&mut ptx, "    .param .u64 %param_a_meta,")?;
    writeln(&mut ptx, "    .param .u64 %param_b,")?;
    writeln(&mut ptx, "    .param .u64 %param_c,")?;
    writeln(&mut ptx, "    .param .u32 %param_m,")?;
    writeln(&mut ptx, "    .param .u32 %param_n,")?;
    writeln(&mut ptx, "    .param .u32 %param_k")?;
    writeln(&mut ptx, ")")?;
    writeln(&mut ptx, "{")?;

    // Register / fragment declarations.
    writeln(&mut ptx, "    .reg .b32 %r<32>;")?;
    writeln(&mut ptx, "    .reg .b64 %rd<16>;")?;
    writeln(&mut ptx, "    .reg .f32 %f<16>;")?;
    // Sparse A fragment (2 x .f16x2 = 4 halves, half the dense 8), dense B
    // fragment (2 x .f16x2), F32 accumulator (4 lanes per thread), metadata.
    writeln(&mut ptx, "    .reg .b32 %ra<2>;   // compressed A fragment")?;
    writeln(&mut ptx, "    .reg .b32 %rb<2>;   // dense B fragment")?;
    writeln(&mut ptx, "    .reg .b32 %rmeta;   // 2:4 lane metadata")?;
    writeln(&mut ptx, "    .reg .f32 %rc<4>;   // C accumulator")?;
    writeln(&mut ptx, "")?;

    // Load kernel parameters.
    writeln(&mut ptx, "    ld.param.u64 %rd0, [%param_a_values];")?;
    writeln(&mut ptx, "    ld.param.u64 %rd1, [%param_a_meta];")?;
    writeln(&mut ptx, "    ld.param.u64 %rd2, [%param_b];")?;
    writeln(&mut ptx, "    ld.param.u64 %rd3, [%param_c];")?;
    writeln(&mut ptx, "    ld.param.u32 %r0, [%param_m];")?;
    writeln(&mut ptx, "    ld.param.u32 %r1, [%param_n];")?;
    writeln(&mut ptx, "    ld.param.u32 %r2, [%param_k];")?;
    writeln(&mut ptx, "")?;

    // Zero the accumulator fragment.
    writeln(&mut ptx, "    mov.f32 %rc0, 0f00000000;")?;
    writeln(&mut ptx, "    mov.f32 %rc1, 0f00000000;")?;
    writeln(&mut ptx, "    mov.f32 %rc2, 0f00000000;")?;
    writeln(&mut ptx, "    mov.f32 %rc3, 0f00000000;")?;
    writeln(&mut ptx, "")?;

    // Load the per-group lane metadata once (selects the live A lanes).
    writeln(&mut ptx, "    // Load 2:4 structured-sparsity metadata")?;
    writeln(&mut ptx, "    ld.global.b32 %rmeta, [%rd1];")?;
    writeln(&mut ptx, "")?;

    // Mainloop over K in steps of 16 dense / 8 compressed.
    writeln(
        &mut ptx,
        &format!("    // Sparse mainloop: {k_iters} iterations of m16n8k16"),
    )?;
    for it in 0..k_iters {
        writeln(&mut ptx, &format!("    // --- k-iter {it} ---"))?;
        // Load compressed A fragment (half the dense bytes).
        writeln(&mut ptx, "    ld.global.b32 %ra0, [%rd0];")?;
        writeln(&mut ptx, "    ld.global.b32 %ra1, [%rd0+4];")?;
        // Load dense B fragment.
        writeln(&mut ptx, "    ld.global.b32 %rb0, [%rd2];")?;
        writeln(&mut ptx, "    ld.global.b32 %rb1, [%rd2+4];")?;
        // Sparse Tensor-Core MMA: C += A_sparse * B using metadata selection.
        writeln(
            &mut ptx,
            "    mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
        )?;
        writeln(&mut ptx, "        { %rc0, %rc1, %rc2, %rc3 },")?;
        writeln(&mut ptx, "        { %ra0, %ra1 },")?;
        writeln(&mut ptx, "        { %rb0, %rb1 },")?;
        writeln(&mut ptx, "        { %rc0, %rc1, %rc2, %rc3 }, %rmeta, 0x0;")?;
        // Advance compressed-A and B pointers by one K-tile.
        writeln(&mut ptx, "    add.s64 %rd0, %rd0, 8;")?;
        writeln(&mut ptx, "    add.s64 %rd2, %rd2, 8;")?;
    }
    writeln(&mut ptx, "")?;

    // Store the accumulator fragment to C.
    writeln(&mut ptx, "    // Store C tile")?;
    writeln(&mut ptx, "    st.global.f32 [%rd3], %rc0;")?;
    writeln(&mut ptx, "    st.global.f32 [%rd3+4], %rc1;")?;
    writeln(&mut ptx, "    st.global.f32 [%rd3+8], %rc2;")?;
    writeln(&mut ptx, "    st.global.f32 [%rd3+12], %rc3;")?;
    writeln(&mut ptx, "")?;
    writeln(&mut ptx, "    ret;")?;
    writeln(&mut ptx, "}")?;

    Ok(ptx)
}

/// Appends `line` followed by a newline to `ptx`.
fn writeln(ptx: &mut String, line: &str) -> BlasResult<()> {
    writeln_impl(ptx, line)
}

#[inline]
fn writeln_impl(ptx: &mut String, line: &str) -> BlasResult<()> {
    ptx.write_str(line)
        .and_then(|()| ptx.write_char('\n'))
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive dense GEMM `C = A * B` (row-major) for cross-checking.
    fn naive_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = a[i * k + p];
                if aip == 0.0 {
                    continue;
                }
                for j in 0..n {
                    c[i * n + j] += aip * b[p * n + j];
                }
            }
        }
        c
    }

    /// Build a dense matrix that already obeys a 2:4 pattern: in each group of
    /// four, lanes 0 and 2 are non-zero, lanes 1 and 3 are zero.
    fn make_2to4_dense(rows: usize, cols: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; rows * cols];
        for r in 0..rows {
            for g in 0..(cols / GROUP) {
                let base = r * cols + g * GROUP;
                v[base] = (r + g + 1) as f32; // lane 0
                v[base + 2] = (r * 2 + g) as f32 + 0.5; // lane 2
            }
        }
        v
    }

    #[test]
    fn nibble_round_trip() {
        for a in 0u8..4 {
            for b in 0u8..4 {
                let m = TwoFourMeta { lanes: [a, b] };
                assert_eq!(TwoFourMeta::from_nibble(m.pack_nibble()), m);
            }
        }
    }

    #[test]
    fn metadata_pack_unpack_round_trip() {
        let meta: Vec<TwoFourMeta> = (0..20)
            .map(|i| TwoFourMeta {
                lanes: [(i % 4) as u8, ((i + 1) % 4).max(i % 4) as u8],
            })
            .collect();
        let words = pack_metadata(&meta);
        // 20 groups -> ceil(20/8) = 3 words.
        assert_eq!(words.len(), 3);
        let back = unpack_metadata(&words, meta.len());
        assert_eq!(back, meta);
    }

    #[test]
    fn compress_decompress_exact_for_2to4_input() {
        // A matrix already in 2:4 form must round-trip exactly.
        let (rows, cols) = (3, 8);
        let dense = make_2to4_dense(rows, cols);
        let c = compress_2to4(&dense, rows, cols).expect("compress");
        assert_eq!(c.values.len(), rows * (cols / 2));
        assert_eq!(c.meta.len(), rows * (cols / 4));
        let back = decompress_2to4(&c).expect("decompress");
        assert_eq!(back, dense);
    }

    #[test]
    fn compress_keeps_two_largest_magnitude() {
        // Group [1, -9, 3, 2]: keep lanes 1 (|−9|) and 2 (|3|), ascending.
        let dense = vec![1.0, -9.0, 3.0, 2.0];
        let c = compress_2to4(&dense, 1, 4).expect("compress");
        assert_eq!(c.meta[0].lanes, [1, 2]);
        // Values stored in ascending lane order: lane1 = -9, lane2 = 3.
        assert_eq!(c.values, vec![-9.0, 3.0]);
        // The dropped lanes (0 and 3) become zero on decompress.
        let back = decompress_2to4(&c).expect("decompress");
        assert_eq!(back, vec![0.0, -9.0, 3.0, 0.0]);
    }

    #[test]
    fn compress_ties_break_to_lower_lane() {
        // Group [5, 5, 5, 5]: all equal magnitude -> keep the two lowest lanes.
        let dense = vec![5.0, 5.0, 5.0, 5.0];
        let c = compress_2to4(&dense, 1, 4).expect("compress");
        assert_eq!(c.meta[0].lanes, [0, 1]);
    }

    #[test]
    fn spgemm_matches_dense_gemm() {
        // SpGEMM on a true 2:4 A must equal a dense GEMM of the same A.
        let (m, k, n) = (4, 8, 5);
        let a_dense = make_2to4_dense(m, k);
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.1) - 1.3).collect();
        let a = compress_2to4(&a_dense, m, k).expect("compress");

        let mut c = vec![0.0_f32; m * n];
        spgemm_2to4(&a, &b, n, 1.0, 0.0, &mut c).expect("spgemm");

        let reference = naive_gemm(&a_dense, &b, m, k, n);
        for (got, want) in c.iter().zip(reference.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn spgemm_matches_pruned_dense_for_arbitrary_input() {
        // For an arbitrary dense A, SpGEMM(compress(A)) must equal a dense GEMM
        // of the *pruned* A (decompress(compress(A))).
        let (m, k, n) = (3, 12, 4);
        let a_dense: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32).sin()).collect();
        let a = compress_2to4(&a_dense, m, k).expect("compress");
        let a_pruned = decompress_2to4(&a).expect("decompress");

        let mut c = vec![0.0_f32; m * n];
        spgemm_2to4(&a, &b, n, 1.0, 0.0, &mut c).expect("spgemm");
        let reference = naive_gemm(&a_pruned, &b, m, k, n);
        for (got, want) in c.iter().zip(reference.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn spgemm_alpha_beta() {
        let (m, k, n) = (2, 4, 2);
        let a_dense = make_2to4_dense(m, k);
        let b: Vec<f32> = (1..=k * n).map(|i| i as f32).collect();
        let a = compress_2to4(&a_dense, m, k).expect("compress");

        // C0 = 1 * A*B
        let mut c0 = vec![0.0_f32; m * n];
        spgemm_2to4(&a, &b, n, 1.0, 0.0, &mut c0).expect("spgemm");

        // C = 2 * A*B + 3 * C_init, with C_init = ones.
        let mut c = vec![1.0_f32; m * n];
        spgemm_2to4(&a, &b, n, 2.0, 3.0, &mut c).expect("spgemm");

        for i in 0..m * n {
            let want = 2.0 * c0[i] + 3.0 * 1.0;
            assert!(
                (c[i] - want).abs() < 1e-4,
                "i={i}: got {}, want {want}",
                c[i]
            );
        }
    }

    #[test]
    fn spgemm_beta_zero_ignores_initial_c() {
        let (m, k, n) = (2, 4, 3);
        let a_dense = make_2to4_dense(m, k);
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 + 1.0).collect();
        let a = compress_2to4(&a_dense, m, k).expect("compress");

        let mut c_nan = vec![f32::NAN; m * n];
        spgemm_2to4(&a, &b, n, 1.0, 0.0, &mut c_nan).expect("spgemm");
        // beta == 0 must clear the NaNs (no read of prior C).
        for &v in &c_nan {
            assert!(v.is_finite(), "NaN leaked through beta=0");
        }
    }

    #[test]
    fn spgemm_identity_b_returns_pruned_a() {
        // B = I_k -> A_sparse * I = pruned A.
        let (m, k) = (3, 8);
        let a_dense = make_2to4_dense(m, k);
        let a = compress_2to4(&a_dense, m, k).expect("compress");
        let a_pruned = decompress_2to4(&a).expect("decompress");
        let mut bi = vec![0.0_f32; k * k];
        for i in 0..k {
            bi[i * k + i] = 1.0;
        }
        let mut c = vec![0.0_f32; m * k];
        spgemm_2to4(&a, &bi, k, 1.0, 0.0, &mut c).expect("spgemm");
        for (got, want) in c.iter().zip(a_pruned.iter()) {
            assert!((got - want).abs() < 1e-5);
        }
    }

    #[test]
    fn compress_errors() {
        assert!(matches!(
            compress_2to4(&[], 0, 4),
            Err(BlasError::InvalidDimension(_))
        ));
        // cols not multiple of 4.
        assert!(matches!(
            compress_2to4(&[1.0; 6], 1, 6),
            Err(BlasError::InvalidArgument(_))
        ));
        // wrong length.
        assert!(matches!(
            compress_2to4(&[1.0; 3], 1, 4),
            Err(BlasError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn spgemm_dim_errors() {
        let a = compress_2to4(&make_2to4_dense(2, 4), 2, 4).expect("compress");
        // B wrong length.
        let mut c = vec![0.0_f32; 2 * 3];
        assert!(matches!(
            spgemm_2to4(&a, &[1.0; 5], 3, 1.0, 0.0, &mut c),
            Err(BlasError::DimensionMismatch(_))
        ));
        // C wrong length.
        let mut c_bad = vec![0.0_f32; 5];
        assert!(matches!(
            spgemm_2to4(&a, &[1.0; 4 * 3], 3, 1.0, 0.0, &mut c_bad),
            Err(BlasError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn config_availability_and_ratio() {
        assert!(!SparseGemmConfig::is_available(SmVersion::Sm75));
        assert!(SparseGemmConfig::is_available(SmVersion::Sm80));
        assert!(SparseGemmConfig::is_available(SmVersion::Sm90));
        assert!((SparseGemmConfig::compression_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn config_validate_rejects_bad_k_and_arch() {
        let cfg = SparseGemmConfig { m: 16, n: 8, k: 18 };
        // k not multiple of 4.
        assert!(matches!(
            cfg.validate(SmVersion::Sm80),
            Err(BlasError::InvalidArgument(_))
        ));
        // pre-Ampere unsupported.
        let cfg_ok = SparseGemmConfig { m: 16, n: 8, k: 16 };
        assert!(matches!(
            cfg_ok.validate(SmVersion::Sm75),
            Err(BlasError::UnsupportedOperation(_))
        ));
        assert!(cfg_ok.validate(SmVersion::Sm80).is_ok());
    }

    #[test]
    fn ptx_contains_sparse_mma_and_metadata() {
        let cfg = SparseGemmConfig { m: 16, n: 8, k: 32 };
        let ptx = generate_sparse_gemm_ptx(&cfg, SmVersion::Sm80).expect("ptx");
        // Sparse MMA instruction present with the correct 16x8x16 shape.
        assert!(ptx.contains("mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
        // Metadata operand loaded and fed to the sparse mma.
        assert!(ptx.contains("%rmeta"));
        assert!(ptx.contains("metadata"));
        // Kernel signature exposes the compressed A + metadata params.
        assert!(ptx.contains("%param_a_values"));
        assert!(ptx.contains("%param_a_meta"));
        // Target + version line for sm_80.
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains(".version 7.0"));
        // Mangled kernel name.
        assert!(ptx.contains("spgemm_2to4_16x8x32"));
    }

    #[test]
    fn ptx_mainloop_iteration_count_tracks_k() {
        // k=32 -> 2 mainloop iterations of m16n8k16; k=16 -> 1.
        let two =
            generate_sparse_gemm_ptx(&SparseGemmConfig { m: 16, n: 8, k: 32 }, SmVersion::Sm80)
                .expect("ptx");
        assert_eq!(two.matches("mma.sp.sync.aligned").count(), 2);

        let one =
            generate_sparse_gemm_ptx(&SparseGemmConfig { m: 16, n: 8, k: 16 }, SmVersion::Sm80)
                .expect("ptx");
        assert_eq!(one.matches("mma.sp.sync.aligned").count(), 1);
    }

    #[test]
    fn ptx_rejects_unsupported_arch() {
        let cfg = SparseGemmConfig { m: 16, n: 8, k: 16 };
        assert!(matches!(
            generate_sparse_gemm_ptx(&cfg, SmVersion::Sm75),
            Err(BlasError::UnsupportedOperation(_))
        ));
    }
}
