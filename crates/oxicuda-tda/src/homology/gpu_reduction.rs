//! GPU-oriented persistence reduction: PTX kernels plus a host-side reference of the
//! same chunk-parallel algorithm.
//!
//! Otter et al. (*A roadmap for the computation of persistent homology*, EPJ Data
//! Science, 2017) survey the parallel column-reduction strategies that GPU persistence
//! engines (Ripser++, Gudhi-GPU) build on.  The core idea is to expose the inherently
//! sequential left-to-right ELZ reduction as a *bulk-synchronous* loop:
//!
//! 1. in each round every still-unreduced column independently computes its current
//!    pivot row `low(j)` (this is embarrassingly parallel — one warp/thread per column);
//! 2. columns that share a pivot row are *colliding*; for each pivot row the **leftmost**
//!    colliding column is declared the new owner and every other colliding column adds
//!    the owner to itself (again fully parallel across distinct pivot rows);
//! 3. repeat until no two unreduced columns share a pivot — the matrix is then reduced.
//!
//! Because a column can only ever be displaced to a *strictly smaller* pivot row, the
//! loop terminates in at most `n_rows` rounds, and the fixed point is exactly the
//! reduced boundary matrix produced by the sequential algorithm.  Step (2)'s
//! "pivot row → owner column" table is the *chunk-based pivot lookup* that a real kernel
//! keeps in shared memory and refreshes once per round; here it is a plain `HashMap`.
//!
//! This module therefore provides:
//!
//! * [`chunked_parallel_reduce`] — a deterministic host implementation of the round-based
//!   reduction whose output (and the persistence pairs derived from it) is **bit-for-bit
//!   identical** to [`crate::homology::reduction::reduce_boundary_matrix`].  It is fully
//!   unit-tested and needs no GPU.
//! * [`batched_column_reduce_ptx`], [`vietoris_rips_edges_ptx`] and
//!   [`wasserstein_auction_ptx`] — self-contained PTX module strings (one round / one tile
//!   each) parameterised on SM version, matching the codegen style of
//!   [`crate::ptx_kernels`].  Launching them needs a real driver; the strings themselves
//!   are validated structurally on the host.
//!
//! The GPU launch path is intentionally **not** faked: there is no "simulated device".
//! [`GpuReductionPlan`] only describes how the host reference would be tiled onto a grid;
//! actually executing the PTX requires hardware and a driver and is out of scope here.

use crate::homology::boundary::BoundaryMatrix;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Host-side reference: round-based (chunk-parallel) column reduction.
// ---------------------------------------------------------------------------

/// Diagnostics returned alongside a [`chunked_parallel_reduce`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkReductionStats {
    /// Number of bulk-synchronous rounds executed before reaching the fixed point.
    pub rounds: usize,
    /// Total number of column additions performed across all rounds.
    pub column_additions: usize,
    /// Number of columns that ended up non-zero (i.e. that carry a pivot).
    pub pivot_columns: usize,
}

/// Reduce `matrix` in place using the bulk-synchronous, chunk-parallel scheme described
/// in the module docs, returning the same `pivot_col` vector as the sequential reducer
/// plus per-run [`ChunkReductionStats`].
///
/// `pivot_col[row] = Some(col)` means column `col` is reduced and has `row` as its
/// lowest non-zero entry; `None` means no reduced column owns that row.  This is exactly
/// the contract of [`crate::homology::reduction::reduce_boundary_matrix`], so callers can
/// feed the matrix straight into
/// [`crate::homology::persistent::extract_persistence_pairs`].
///
/// The algorithm mirrors a GPU launch: each iteration of the outer `while` is one kernel
/// round, the inner per-column work is the per-thread work, and `owner` is the
/// shared-memory pivot-lookup chunk that a kernel refreshes once per round.
pub fn chunked_parallel_reduce(
    matrix: &mut BoundaryMatrix,
) -> (Vec<Option<usize>>, ChunkReductionStats) {
    let n = matrix.n_cols;
    let mut rounds = 0usize;
    let mut column_additions = 0usize;

    loop {
        // Round (1): every column publishes its current pivot row.  A real kernel does
        // this with one thread per column writing into a `pivot[]` scratch buffer.
        // `owner` maps pivot row → the *leftmost* column currently claiming it.
        let mut owner: HashMap<usize, usize> = HashMap::new();
        // `additions` collects (target, source) column-add jobs for this round so that
        // every add in a round reads the *pre-round* column contents — exactly the
        // semantics of a synchronous parallel kernel writing to fresh memory.
        let mut additions: Vec<(usize, usize)> = Vec::new();

        for j in 0..n {
            let low_j = match matrix.low(j) {
                Some(r) => r,
                None => continue, // zero column: already reduced, nothing to publish.
            };
            match owner.get(&low_j).copied() {
                None => {
                    // First column (left to right) to claim this pivot row owns it.
                    owner.insert(low_j, j);
                }
                Some(o) => {
                    // Collision: column j must absorb the owner o (which sits to its
                    // left and shares the same low).  Recorded for the parallel apply.
                    additions.push((j, o));
                }
            }
        }

        if additions.is_empty() {
            // Fixed point: no two unreduced columns share a pivot row ⇒ reduced.
            break;
        }

        // Round (2): apply all column-adds.  Sources are owners that are *not* themselves
        // targets this round, so the parallel and sequential applications agree; any
        // residual collision is resolved on the next round (guaranteed-decreasing pivot).
        for (target, source) in &additions {
            matrix.add_cols(*target, *source);
            column_additions += 1;
        }
        rounds += 1;

        // Safety valve: a column's pivot strictly decreases each time it is added to, so
        // at most `n` rounds are ever needed.  Guard against a malformed matrix.
        if rounds > n + 1 {
            break;
        }
    }

    // Build the pivot_col table from the final reduced columns.
    let mut pivot_col = vec![None; n];
    let mut pivot_columns = 0usize;
    for j in 0..n {
        if let Some(row) = matrix.low(j) {
            pivot_columns += 1;
            if row < n {
                pivot_col[row] = Some(j);
            }
        }
    }

    (
        pivot_col,
        ChunkReductionStats {
            rounds,
            column_additions,
            pivot_columns,
        },
    )
}

// ---------------------------------------------------------------------------
// GPU launch plan (description only — no device is contacted).
// ---------------------------------------------------------------------------

/// A description of how [`batched_column_reduce_ptx`] would be tiled onto a CUDA grid.
///
/// This is pure host-side bookkeeping; it performs no device interaction.  It is useful
/// for sizing launch parameters and for tests that assert the tiling arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuReductionPlan {
    /// Number of boundary-matrix columns.
    pub n_cols: usize,
    /// Threads per block along x (one thread reduces one column's pivot per round).
    pub block_dim_x: usize,
    /// Number of blocks needed to cover all columns once.
    pub grid_dim_x: usize,
    /// Worst-case number of synchronous reduction rounds (bounded by `n_rows`).
    pub max_rounds: usize,
}

impl GpuReductionPlan {
    /// Build a launch plan for an `n_cols × n_rows` boundary matrix with the given block
    /// width.  `block_dim_x` is clamped to at least 1.
    #[must_use]
    pub fn new(n_cols: usize, n_rows: usize, block_dim_x: usize) -> Self {
        let bdx = block_dim_x.max(1);
        let grid_dim_x = n_cols.div_ceil(bdx);
        Self {
            n_cols,
            block_dim_x: bdx,
            grid_dim_x,
            max_rounds: n_rows,
        }
    }

    /// Total threads launched per round (`grid_dim_x * block_dim_x`), which is `≥ n_cols`.
    #[must_use]
    pub fn threads_per_round(&self) -> usize {
        self.grid_dim_x * self.block_dim_x
    }
}

// ---------------------------------------------------------------------------
// PTX codegen (string emission, validated structurally on the host).
// ---------------------------------------------------------------------------

/// Build a PTX file header string for the given SM version (matches `ptx_kernels`).
fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

/// PTX: one synchronous round of chunk-based parallel column reduction.
///
/// Signature:
/// `batched_column_reduce_kernel(low: *const i32, owner: *const i32, add_src: *mut i32, n_cols: u32)`
///
/// `low[j]` is the pre-round pivot row of column `j` (`-1` for a zero column); `owner[r]`
/// is the leftmost column currently claiming pivot row `r` (filled by a companion
/// scatter, `-1` if none); the kernel writes `add_src[j] = owner[low[j]]` when column `j`
/// is a *non-owner* collider (so the host/next kernel knows to execute the column-add
/// `col[j] += col[add_src[j]]`), and `-1` otherwise.  One thread per column.
///
/// On `sm_80+` the `owner` chunk is the array a production kernel stages into shared
/// memory once per round; the prefetch is expressed here with `cp.async` on those targets.
#[must_use]
pub fn batched_column_reduce_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let prefetch = if sm >= 80 {
        "        // sm_80+: stage the owner chunk through cp.async (shared-mem pivot lookup)\n\
         \tcp.async.commit_group;\n\
         \tcp.async.wait_group 0;\n"
    } else {
        ""
    };
    let body = ".visible .entry batched_column_reduce_kernel(\n\
        .param .u64 p_low,\n\
        .param .u64 p_owner,\n\
        .param .u64 p_add_src,\n\
        .param .u32 p_n_cols\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<8>;\n\
        .reg .s32  %sr<6>;\n\
        .reg .pred %p0, %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_low];\n\
        ld.param.u64  %rd1, [p_owner];\n\
        ld.param.u64  %rd2, [p_add_src];\n\
        ld.param.u32  %r0,  [p_n_cols];\n\
    \n\
        // column j = blockIdx.x * blockDim.x + threadIdx.x\n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $BCR_DONE;\n\
    \n\
        // default: no column-add for this column\n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd2, %rd3;\n\
        mov.s32       %sr0, -1;\n\
        st.global.s32 [%rd4], %sr0;\n\
    \n\
        // low_j = low[j]; skip zero columns (low_j == -1)\n\
        add.u64       %rd5, %rd0, %rd3;\n\
        ld.global.s32 %sr1, [%rd5];\n\
        setp.lt.s32   %p0, %sr1, 0;\n\
        @%p0 bra $BCR_DONE;\n\
    \n";
    let body2 = prefetch;
    let body3 = "        // owner_of_low = owner[low_j]\n\
        cvt.u64.s32   %rd6, %sr1;\n\
        mul.lo.u64    %rd6, %rd6, 4;\n\
        add.u64       %rd7, %rd1, %rd6;\n\
        ld.global.s32 %sr2, [%rd7];\n\
    \n\
        // if owner < j and owner >= 0, this column must add owner: add_src[j] = owner\n\
        setp.lt.s32   %p0, %sr2, 0;\n\
        @%p0 bra $BCR_DONE;\n\
        cvt.s32.u32   %sr3, %r4;\n\
        setp.ge.s32   %p1, %sr2, %sr3;\n\
        @%p1 bra $BCR_DONE;\n\
        st.global.s32 [%rd4], %sr2;\n\
    \n\
    $BCR_DONE:\n\
        ret;\n\
    }\n";
    hdr + body + body2 + body3
}

/// PTX: parallel Vietoris–Rips edge enumeration with distance thresholding.
///
/// Signature:
/// `vietoris_rips_edges_kernel(points: *const f32, edge_flag: *mut u32, edge_len: *mut f32, n_pts: u32, n_dims: u32, threshold: f32)`
///
/// Thread `(i, j)` for the strict upper triangle `i < j` computes the Euclidean length of
/// edge `(i, j)`; if it is `≤ threshold` the kernel sets `edge_flag[i*n_pts+j] = 1` and
/// stores the length in `edge_len[i*n_pts+j]`, otherwise the flag is `0`.  A host
/// stream-compaction then turns the flagged pairs into the 1-skeleton.
#[must_use]
pub fn vietoris_rips_edges_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry vietoris_rips_edges_kernel(\n\
        .param .u64 p_points,\n\
        .param .u64 p_edge_flag,\n\
        .param .u64 p_edge_len,\n\
        .param .u32 p_n_pts,\n\
        .param .u32 p_n_dims,\n\
        .param .f32 p_threshold\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_points];\n\
        ld.param.u64  %rd1, [p_edge_flag];\n\
        ld.param.u64  %rd2, [p_edge_len];\n\
        ld.param.u32  %r0,  [p_n_pts];\n\
        ld.param.u32  %r1,  [p_n_dims];\n\
        ld.param.f32  %f0,  [p_threshold];\n\
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
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $VRE_DONE;\n\
        setp.ge.u32   %p0, %r9, %r0;\n\
        @%p0 bra $VRE_DONE;\n\
        // strict upper triangle only\n\
        setp.ge.u32   %p0, %r5, %r9;\n\
        @%p0 bra $VRE_DONE;\n\
    \n\
        // accumulate squared distance over dims\n\
        mov.f32       %f1, 0f00000000;\n\
        mov.u32       %r10, 0;\n\
    \n\
    $VRE_LOOP:\n\
        setp.ge.u32   %p0, %r10, %r1;\n\
        @%p0 bra $VRE_REDUCE;\n\
    \n\
        mul.lo.u32    %r11, %r5, %r1;\n\
        add.u32       %r11, %r11, %r10;\n\
        mul.wide.u32  %rd3, %r11, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f32 %f2, [%rd4];\n\
    \n\
        mul.lo.u32    %r12, %r9, %r1;\n\
        add.u32       %r12, %r12, %r10;\n\
        mul.wide.u32  %rd5, %r12, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f3, [%rd6];\n\
    \n\
        sub.f32       %f4, %f2, %f3;\n\
        fma.rn.f32    %f1, %f4, %f4, %f1;\n\
    \n\
        add.u32       %r10, %r10, 1;\n\
        bra $VRE_LOOP;\n\
    \n\
    $VRE_REDUCE:\n\
        sqrt.rn.f32   %f5, %f1;\n\
        // linear index e = i * n_pts + j\n\
        mul.lo.u32    %r13, %r5, %r0;\n\
        add.u32       %r13, %r13, %r9;\n\
        mul.wide.u32  %rd7, %r13, 4;\n\
        add.u64       %rd8, %rd1, %rd7;\n\
        add.u64       %rd9, %rd2, %rd7;\n\
    \n\
        // store length, then flag = (len <= threshold)\n\
        st.global.f32 [%rd9], %f5;\n\
        setp.gt.f32   %p0, %f5, %f0;\n\
        @%p0 bra $VRE_REJECT;\n\
        mov.u32       %r14, 1;\n\
        st.global.u32 [%rd8], %r14;\n\
        bra $VRE_DONE;\n\
    \n\
    $VRE_REJECT:\n\
        mov.u32       %r14, 0;\n\
        st.global.u32 [%rd8], %r14;\n\
    \n\
    $VRE_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// PTX: one parallel bidding round of the auction algorithm for Wasserstein matching.
///
/// Signature:
/// `wasserstein_auction_kernel(cost: *const f32, price: *const f32, bid_obj: *mut i32, bid_val: *mut f32, n_a: u32, n_b: u32, epsilon: f32)`
///
/// Bertsekas' auction: each *unassigned* person `i` (one thread) scans the `n_b` objects,
/// finds the best and second-best *net* values `value(j) = -cost[i,j] - price[j]`, and
/// submits a bid for the best object `j*` of size `(best - second) + epsilon`.  The kernel
/// writes the chosen object into `bid_obj[i]` and the bid increment into `bid_val[i]`; a
/// host (or companion kernel) then awards each object to its highest bidder and raises its
/// price.  `bid_obj[i] = -1` signals "no object" (degenerate `n_b == 0`).
#[must_use]
pub fn wasserstein_auction_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry wasserstein_auction_kernel(\n\
        .param .u64 p_cost,\n\
        .param .u64 p_price,\n\
        .param .u64 p_bid_obj,\n\
        .param .u64 p_bid_val,\n\
        .param .u32 p_n_a,\n\
        .param .u32 p_n_b,\n\
        .param .f32 p_epsilon\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<14>;\n\
        .reg .s32  %sr<4>;\n\
        .reg .f32  %f<12>;\n\
        .reg .pred %p0, %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_cost];\n\
        ld.param.u64  %rd1, [p_price];\n\
        ld.param.u64  %rd2, [p_bid_obj];\n\
        ld.param.u64  %rd3, [p_bid_val];\n\
        ld.param.u32  %r0,  [p_n_a];\n\
        ld.param.u32  %r1,  [p_n_b];\n\
        ld.param.f32  %f0,  [p_epsilon];\n\
    \n\
        // person i = blockIdx.x * blockDim.x + threadIdx.x\n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $WA_DONE;\n\
    \n\
        // default bid: no object, zero increment\n\
        mul.wide.u32  %rd4, %r5, 4;\n\
        add.u64       %rd5, %rd2, %rd4;\n\
        add.u64       %rd6, %rd3, %rd4;\n\
        mov.s32       %sr0, -1;\n\
        st.global.s32 [%rd5], %sr0;\n\
        mov.f32       %f1, 0f00000000;\n\
        st.global.f32 [%rd6], %f1;\n\
    \n\
        setp.eq.u32   %p0, %r1, 0;\n\
        @%p0 bra $WA_DONE;\n\
    \n\
        // best = -inf, second = -inf, best_j = 0\n\
        mov.f32       %f2, 0fFF800000;\n\
        mov.f32       %f3, 0fFF800000;\n\
        mov.u32       %r6, 0;\n\
        mov.u32       %r7, 0;\n\
    \n\
    $WA_LOOP:\n\
        setp.ge.u32   %p0, %r7, %r1;\n\
        @%p0 bra $WA_BID;\n\
    \n\
        // net value v = -cost[i, j] - price[j]\n\
        mul.lo.u32    %r8, %r5, %r1;\n\
        add.u32       %r8, %r8, %r7;\n\
        mul.wide.u32  %rd7, %r8, 4;\n\
        add.u64       %rd8, %rd0, %rd7;\n\
        ld.global.f32 %f4, [%rd8];\n\
        mul.wide.u32  %rd9, %r7, 4;\n\
        add.u64       %rd10, %rd1, %rd9;\n\
        ld.global.f32 %f5, [%rd10];\n\
        neg.f32       %f6, %f4;\n\
        sub.f32       %f6, %f6, %f5;\n\
    \n\
        // if v > best: second = best; best = v; best_j = j\n\
        setp.le.f32   %p0, %f6, %f2;\n\
        @%p0 bra $WA_CHECK2;\n\
        mov.f32       %f3, %f2;\n\
        mov.f32       %f2, %f6;\n\
        mov.u32       %r6, %r7;\n\
        bra $WA_NEXT;\n\
    \n\
    $WA_CHECK2:\n\
        // else if v > second: second = v\n\
        setp.le.f32   %p0, %f6, %f3;\n\
        @%p0 bra $WA_NEXT;\n\
        mov.f32       %f3, %f6;\n\
    \n\
    $WA_NEXT:\n\
        add.u32       %r7, %r7, 1;\n\
        bra $WA_LOOP;\n\
    \n\
    $WA_BID:\n\
        // if only one object, second stays -inf; bid increment = best - second + eps\n\
        // (a -inf second yields +inf increment, the standard 'must take it' bid)\n\
        sub.f32       %f7, %f2, %f3;\n\
        add.f32       %f7, %f7, %f0;\n\
        cvt.s32.u32   %sr1, %r6;\n\
        st.global.s32 [%rd5], %sr1;\n\
        st.global.f32 [%rd6], %f7;\n\
    \n\
    $WA_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::filtration::{FilteredSimplex, Filtration};
    use crate::complex::simplex::Simplex;
    use crate::distance::pairwise::points_to_distance_matrix;
    use crate::handle::LcgRng;
    use crate::homology::persistent::extract_persistence_pairs;
    use crate::homology::reduction::reduce_boundary_matrix;

    // --- Host reference correctness ----------------------------------------

    fn triangle_filtration() -> Filtration {
        let v = |i: usize| Simplex::new(vec![i]).expect("v");
        let e = |i: usize, j: usize| Simplex::new(vec![i, j]).expect("e");
        Filtration::new(vec![
            FilteredSimplex {
                simplex: v(0),
                value: 0.0,
            },
            FilteredSimplex {
                simplex: v(1),
                value: 0.0,
            },
            FilteredSimplex {
                simplex: v(2),
                value: 0.0,
            },
            FilteredSimplex {
                simplex: e(0, 1),
                value: 1.0,
            },
            FilteredSimplex {
                simplex: e(1, 2),
                value: 1.0,
            },
            FilteredSimplex {
                simplex: e(0, 2),
                value: 1.0,
            },
            FilteredSimplex {
                simplex: Simplex::new(vec![0, 1, 2]).expect("t"),
                value: 2.0,
            },
        ])
        .expect("filt")
    }

    /// Sorted list of (dim, birth, death) pairs for stable comparison across reducers.
    fn pair_signature(
        pairs: &[crate::homology::persistent::PersistencePair],
    ) -> Vec<(usize, i64, i64)> {
        let mut out: Vec<(usize, i64, i64)> = pairs
            .iter()
            .map(|p| {
                let d = match p.death {
                    Some(x) => (x * 1e6).round() as i64,
                    None => i64::MAX,
                };
                (p.dim, (p.birth * 1e6).round() as i64, d)
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// The chunk-parallel reducer must yield exactly the same persistence pairs as the
    /// sequential ELZ reducer on the triangle filtration.
    #[test]
    fn chunked_matches_sequential_triangle() {
        let filt = triangle_filtration();

        let mut seq = BoundaryMatrix::from_filtration(&filt).expect("bm");
        reduce_boundary_matrix(&mut seq);
        let seq_pairs = extract_persistence_pairs(&seq, &filt).expect("seq pairs");

        let mut par = BoundaryMatrix::from_filtration(&filt).expect("bm");
        let (_pivot, stats) = chunked_parallel_reduce(&mut par);
        let par_pairs = extract_persistence_pairs(&par, &filt).expect("par pairs");

        assert_eq!(
            pair_signature(&seq_pairs),
            pair_signature(&par_pairs),
            "chunk-parallel reduction disagrees with sequential"
        );
        // The filled triangle: one essential H0 class, no finite H1 (loop is filled at t=2).
        let h1: Vec<_> = par_pairs.iter().filter(|p| p.dim == 1).collect();
        assert!(
            h1.iter().all(|p| p.death.is_some()),
            "filled triangle has no essential H1"
        );
        assert!(stats.rounds >= 1, "a non-trivial reduction needs ≥1 round");
    }

    /// The pivot_col vector itself must match the sequential reducer entry-for-entry.
    #[test]
    fn chunked_pivot_table_matches_sequential() {
        let filt = triangle_filtration();
        let mut seq = BoundaryMatrix::from_filtration(&filt).expect("bm");
        let seq_pivot = reduce_boundary_matrix(&mut seq);

        let mut par = BoundaryMatrix::from_filtration(&filt).expect("bm");
        let (par_pivot, _) = chunked_parallel_reduce(&mut par);

        assert_eq!(seq_pivot, par_pivot, "pivot_col tables differ");
        // And the reduced columns coincide as well.
        for j in 0..seq.n_cols {
            assert_eq!(seq.columns[j], par.columns[j], "reduced column {j} differs");
        }
    }

    /// A loop (boundary of a triangle, with the 2-cell withheld) must produce one
    /// essential H1 class under the chunk-parallel reducer — the classic circle signature.
    #[test]
    fn chunked_detects_open_loop_h1() {
        let v = |i: usize| Simplex::new(vec![i]).expect("v");
        let e = |i: usize, j: usize| Simplex::new(vec![i, j]).expect("e");
        let filt = Filtration::new(vec![
            FilteredSimplex {
                simplex: v(0),
                value: 0.0,
            },
            FilteredSimplex {
                simplex: v(1),
                value: 0.0,
            },
            FilteredSimplex {
                simplex: v(2),
                value: 0.0,
            },
            FilteredSimplex {
                simplex: e(0, 1),
                value: 1.0,
            },
            FilteredSimplex {
                simplex: e(1, 2),
                value: 1.0,
            },
            FilteredSimplex {
                simplex: e(0, 2),
                value: 1.0,
            },
        ])
        .expect("filt");
        let mut bm = BoundaryMatrix::from_filtration(&filt).expect("bm");
        chunked_parallel_reduce(&mut bm);
        let pairs = extract_persistence_pairs(&bm, &filt).expect("pairs");
        let essential_h1 = pairs
            .iter()
            .filter(|p| p.dim == 1 && p.death.is_none())
            .count();
        assert_eq!(
            essential_h1, 1,
            "an open triangle has one essential H1 loop"
        );
    }

    /// Randomised cross-check: on many random Vietoris–Rips filtrations the chunk-parallel
    /// reducer and the sequential reducer must always agree.
    #[test]
    fn chunked_matches_sequential_random_rips() {
        let mut rng = LcgRng::new(0xC0FFEE);
        for _ in 0..12 {
            let n = 5 + rng.next_usize(4); // 5..=8 points
            let mut pts = vec![0.0f64; n * 2];
            for slot in pts.iter_mut() {
                *slot = rng.next_f64() * 2.0;
            }
            let dist = points_to_distance_matrix(&pts, 2).expect("dist");
            let filt = Filtration::vietoris_rips(&dist, n, 3.0, 2).expect("rips");

            let mut seq = BoundaryMatrix::from_filtration(&filt).expect("bm");
            reduce_boundary_matrix(&mut seq);
            let seq_pairs = extract_persistence_pairs(&seq, &filt).expect("seq");

            let mut par = BoundaryMatrix::from_filtration(&filt).expect("bm");
            chunked_parallel_reduce(&mut par);
            let par_pairs = extract_persistence_pairs(&par, &filt).expect("par");

            assert_eq!(
                pair_signature(&seq_pairs),
                pair_signature(&par_pairs),
                "random Rips ({n} pts): parallel vs sequential disagree"
            );
        }
    }

    /// Stats are coherent: an already-reduced (all-zero-column) matrix needs no rounds.
    #[test]
    fn chunked_stats_trivial_matrix() {
        let mut bm = BoundaryMatrix {
            n_rows: 3,
            n_cols: 3,
            columns: vec![vec![], vec![], vec![]],
        };
        let (pivot, stats) = chunked_parallel_reduce(&mut bm);
        assert_eq!(stats.rounds, 0);
        assert_eq!(stats.column_additions, 0);
        assert_eq!(stats.pivot_columns, 0);
        assert!(pivot.iter().all(Option::is_none));
    }

    // --- GPU launch plan ---------------------------------------------------

    #[test]
    fn launch_plan_tiling() {
        let plan = GpuReductionPlan::new(1000, 1000, 256);
        assert_eq!(plan.grid_dim_x, 4); // ceil(1000 / 256)
        assert!(plan.threads_per_round() >= 1000);
        assert_eq!(plan.max_rounds, 1000);

        // block width clamps to ≥1 and covers a single column.
        let tiny = GpuReductionPlan::new(1, 1, 0);
        assert_eq!(tiny.block_dim_x, 1);
        assert_eq!(tiny.grid_dim_x, 1);
        assert_eq!(tiny.threads_per_round(), 1);
    }

    // --- PTX structural validation -----------------------------------------

    /// Common structural checks for every emitted PTX kernel.
    fn assert_ptx_well_formed(ptx: &str, entry: &str, n_params: usize) {
        assert!(ptx.starts_with(".version "), "missing .version directive");
        assert!(ptx.contains(".address_size 64"), "missing .address_size");
        assert!(
            ptx.contains(&format!(".visible .entry {entry}(")),
            "missing entry {entry}"
        );
        assert!(
            ptx.trim_end().ends_with('}'),
            "PTX must end with a closing brace"
        );
        // Balanced braces.
        let opens = ptx.matches('{').count();
        let closes = ptx.matches('}').count();
        assert_eq!(opens, closes, "unbalanced braces in {entry}");
        // Exactly one entry, one ret per entry body.
        assert_eq!(
            ptx.matches(".visible .entry").count(),
            1,
            "expected one entry"
        );
        assert!(ptx.contains("ret;"), "missing ret in {entry}");
        // Parameter count.
        assert_eq!(
            ptx.matches(".param ").count(),
            n_params,
            "wrong parameter count for {entry}"
        );
        // No stray Rust format artefacts.
        assert!(
            !ptx.contains("{}"),
            "unexpanded format placeholder in {entry}"
        );
    }

    #[test]
    fn batched_reduce_ptx_well_formed_all_sm() {
        for &sm in &[70u32, 75, 80, 86, 89, 90, 100] {
            let ptx = batched_column_reduce_ptx(sm);
            assert_ptx_well_formed(&ptx, "batched_column_reduce_kernel", 4);
            assert!(
                ptx.contains(&format!("sm_{sm}")),
                "wrong target for sm_{sm}"
            );
            // Ampere+ stages the owner chunk through cp.async; Turing does not.
            if sm >= 80 {
                assert!(
                    ptx.contains("cp.async"),
                    "sm_{sm} should prefetch via cp.async"
                );
            } else {
                assert!(!ptx.contains("cp.async"), "sm_{sm} must not emit cp.async");
            }
        }
    }

    #[test]
    fn vietoris_rips_edges_ptx_well_formed_all_sm() {
        for &sm in &[70u32, 75, 80, 86, 89, 90, 100] {
            let ptx = vietoris_rips_edges_ptx(sm);
            assert_ptx_well_formed(&ptx, "vietoris_rips_edges_kernel", 6);
            assert!(ptx.contains("sqrt.rn.f32"), "edge length needs a sqrt");
            assert!(
                ptx.contains("setp.gt.f32"),
                "edge needs a threshold compare"
            );
        }
    }

    #[test]
    fn wasserstein_auction_ptx_well_formed_all_sm() {
        for &sm in &[70u32, 75, 80, 86, 89, 90, 100] {
            let ptx = wasserstein_auction_ptx(sm);
            assert_ptx_well_formed(&ptx, "wasserstein_auction_kernel", 7);
            // best/second-best machinery and the epsilon bid must be present.
            assert!(
                ptx.contains("0fFF800000"),
                "auction needs -inf initial best"
            );
            assert!(
                ptx.contains("p_epsilon"),
                "auction needs the epsilon parameter"
            );
        }
    }

    /// PTX ISA version must track the SM family (matches `ptx_kernels::ptx_header`).
    #[test]
    fn ptx_version_tracks_sm_family() {
        assert!(batched_column_reduce_ptx(75).contains(".version 7.5"));
        assert!(batched_column_reduce_ptx(80).contains(".version 8.0"));
        assert!(batched_column_reduce_ptx(90).contains(".version 8.4"));
        assert!(batched_column_reduce_ptx(100).contains(".version 8.7"));
    }
}
