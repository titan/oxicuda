//! Load-balanced (merge-based) sparse matrix-vector multiply for CSR graphs.
//!
//! The naive row-parallel SpMV assigns one processor (CPU thread / GPU warp) to
//! each matrix row. On graphs with a **highly skewed degree distribution** — the
//! norm for scale-free / power-law networks such as OGB-products — this leaves
//! most processors idle while a few high-degree rows dominate the runtime. The
//! *merge-based* (a.k.a. *merge-path*) decomposition of Merrill & Garland
//! ("Merge-Based Parallel Sparse Matrix-Vector Multiplication", SC 2016) instead
//! load-balances by the **total amount of work** — `n_rows + n_nonzeros` — split
//! evenly across `p` virtual processors regardless of how the nonzeros cluster.
//!
//! # The merge-path decomposition
//!
//! Think of two strictly increasing sequences:
//!
//! * the *row-pointer* list `A = row_ptr[1 .. n+1]` (one entry per row, marking
//!   where that row ends in the value stream), and
//! * the *natural-number* list `B = [0, 1, …, nnz-1]` (one entry per nonzero).
//!
//! Their order-merge traces a monotone lattice path from `(0, 0)` to
//! `(n, nnz)`: a **down**-step consumes a row boundary (advance in `A`), a
//! **right**-step consumes a nonzero (advance in `B`). The path has exactly
//! `n + nnz` steps, and crucially every unit of SpMV work — emitting a row's
//! accumulator (a down-step) or accumulating one nonzero (a right-step) — maps to
//! exactly one step. Cutting the path into `p` equal-length contiguous segments
//! therefore gives each processor an (almost) identical workload, *independent*
//! of the degree distribution.
//!
//! Each segment's start `(i, j)` (row `i`, nonzero `j`) is found by a binary
//! search along the anti-diagonal `i + j = diag` for the *merge frontier*: the
//! largest `i` with `row_ptr[i] <= j` where `j = diag - i`. This is the classic
//! "find the split point on a 2-D merge path" trick and needs only
//! `O(log n)` per processor.
//!
//! # Carry-out fix-up
//!
//! A row's nonzeros may straddle a segment boundary, so a processor can finish a
//! partial row. Each processor records a *carry-out* — the row index it was in
//! when its segment ended, plus the partial accumulator — and a serial fix-up
//! pass adds these carries into the owning row. This is exactly the GPU
//! reduce-then-scan epilogue, reproduced deterministically on the CPU so the
//! result is **bit-stable** and can be validated against the naive
//! [`crate::graph::csr::CsrGraph::spmv`].
//!
//! The public surface mirrors `CsrGraph::spmv` (dense feature columns) so the
//! balanced path is a drop-in replacement for skewed graphs.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

/// Per-processor carry-out produced by an incomplete trailing row.
///
/// When a segment ends in the middle of a row, `row` is that row index and
/// `partial` holds the `feat_dim`-wide accumulator for the nonzeros the segment
/// consumed but did not get to emit. The serial fix-up adds it into `y[row]`.
#[derive(Debug, Clone)]
struct CarryOut {
    row: usize,
    partial: Vec<f32>,
}

/// Locate the merge-path coordinate `(i, j)` at anti-diagonal index `diag`.
///
/// Returns the number of *down*-steps `i` (rows finished) taken to reach
/// anti-diagonal `diag`; the companion `j = diag - i` is the number of
/// *right*-steps (nonzeros consumed). This is the canonical CUB
/// `MergePathSearch` over list `A[k] = row_ptr[k+1]` (row-end offsets) and
/// `B[k] = k` (the natural numbers): a down-step consumes `A[i]` when
/// `A[i] <= B[j-1]`, i.e. `row_ptr[i+1] <= diag - i - 1`.
///
/// `nnz` is the total nonzero count (length of `B`); `i_hi` is the row upper
/// bound (`n`). The search runs in `O(log n)`.
#[inline]
fn merge_path_search(diag: usize, row_ptr: &[usize], nnz: usize, i_hi: usize) -> usize {
    // Feasible down-step count is clamped so that both `i` and `j = diag - i`
    // stay within their lists: `i ∈ [max(0, diag - nnz), min(diag, n)]`.
    let mut lo = diag.saturating_sub(nnz);
    let mut hi = i_hi.min(diag);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        // B[j - 1] with j = diag - mid ⇒ index (diag - mid - 1); the down-step
        // predicate `row_ptr[i+1] <= diag - i - 1` is written `< diag - i`.
        // mid < hi <= diag ⇒ diag - mid >= 1, so the subtraction never wraps.
        if row_ptr[mid + 1] < diag - mid {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Number of merge segments (virtual processors) for a given problem size.
///
/// Mirrors a GPU launch heuristic: roughly one segment per `target_work` units
/// of `(rows + nnz)`, clamped to `[1, max_processors]`. The result never exceeds
/// the total work, so empty segments are impossible.
#[inline]
fn segment_count(total_work: usize, target_work: usize, max_processors: usize) -> usize {
    if total_work == 0 {
        return 1;
    }
    let by_work = total_work.div_ceil(target_work.max(1));
    by_work.clamp(1, max_processors.max(1)).min(total_work)
}

/// Configuration for the balanced SpMV.
#[derive(Debug, Clone)]
pub struct BalancedSpmvConfig {
    /// Target `(rows + nnz)` work units assigned to each virtual processor.
    ///
    /// Smaller values create more, finer segments (better balance, more
    /// fix-up overhead). Must be `>= 1`.
    pub items_per_segment: usize,
    /// Upper bound on the number of virtual processors.
    ///
    /// Models the finite number of resident warps/CTAs on a real device.
    /// Must be `>= 1`.
    pub max_processors: usize,
}

impl BalancedSpmvConfig {
    /// Construct a configuration, validating both fields are positive.
    pub fn new(items_per_segment: usize, max_processors: usize) -> GnnResult<Self> {
        if items_per_segment == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "items_per_segment must be > 0".to_string(),
            ));
        }
        if max_processors == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "max_processors must be > 0".to_string(),
            ));
        }
        Ok(Self {
            items_per_segment,
            max_processors,
        })
    }
}

impl Default for BalancedSpmvConfig {
    fn default() -> Self {
        // 256 work units/segment, up to 1024 processors — a reasonable proxy for
        // a single mid-range SM occupancy without depending on real hardware.
        Self {
            items_per_segment: 256,
            max_processors: 1024,
        }
    }
}

/// Merge-based load-balanced SpMV: `y = A · X`, `X` is `[n_nodes × feat_dim]`.
///
/// Numerically equivalent (bit-stable, same summation order per row) to
/// [`CsrGraph::spmv`], but the work is partitioned by the merge-path
/// decomposition so skewed degree distributions do not serialise behind a few
/// dense rows.
///
/// # Errors
///
/// * [`GnnError::DimensionMismatch`] if `x.len() != n_nodes * feat_dim`.
pub fn balanced_spmv(
    graph: &CsrGraph,
    x: &[f32],
    feat_dim: usize,
    config: &BalancedSpmvConfig,
) -> GnnResult<Vec<f32>> {
    let n = graph.n_nodes();
    if x.len() != n * feat_dim {
        return Err(GnnError::DimensionMismatch {
            expected: n * feat_dim,
            got: x.len(),
        });
    }

    let row_ptr = graph.row_ptr();
    let col_idx = graph.col_idx();
    let edge_weight = graph.edge_weight();
    let nnz = graph.n_edges();

    let mut y = vec![0.0_f32; n * feat_dim];
    if feat_dim == 0 {
        return Ok(y);
    }

    // Total merge-path length is rows + nonzeros.
    let total_work = n + nnz;
    let n_seg = segment_count(total_work, config.items_per_segment, config.max_processors);
    let mut carries: Vec<CarryOut> = Vec::with_capacity(n_seg);

    for s in 0..n_seg {
        // Inclusive-lo / exclusive-hi anti-diagonal indices for this segment.
        let diag_start = (s * total_work) / n_seg;
        let diag_end = ((s + 1) * total_work) / n_seg;
        if diag_start >= diag_end {
            continue;
        }

        // Resolve both endpoints to merge-path coordinates.
        let i_start = merge_path_search(diag_start, row_ptr, nnz, n);
        let j_start = diag_start - i_start;
        let i_end = merge_path_search(diag_end, row_ptr, nnz, n);
        let j_end = diag_end - i_end;

        // Walk this segment's slice of the merge path. The accumulator holds the
        // nonzeros consumed for the *current* row `i` that have not yet been
        // emitted; it is reset to zero on every row-boundary (down-step) emit.
        let mut i = i_start;
        let mut j = j_start;
        let mut acc = vec![0.0_f32; feat_dim];

        while i < i_end || j < j_end {
            if i < n && j < row_ptr[i + 1] {
                // Right-step: consume nonzero `j` into the current row's acc.
                let col = col_idx[j];
                let w = edge_weight[j];
                let base = col * feat_dim;
                for (k, a) in acc.iter_mut().enumerate() {
                    *a += w * x[base + k];
                }
                j += 1;
            } else {
                // Down-step: row `i`'s boundary lies inside this segment. Emit the
                // accumulated partial (`+=`, so contributions from any earlier
                // segment that also touched row `i` sum correctly) and advance.
                let out_base = i * feat_dim;
                for (k, a) in acc.iter().enumerate() {
                    y[out_base + k] += *a;
                }
                for a in acc.iter_mut() {
                    *a = 0.0;
                }
                i += 1;
            }
        }

        // A non-zero accumulator here means the segment ended partway through row
        // `i` without reaching its boundary: hand that trailing partial to the
        // serial fix-up, which folds it into the owning row. (`i == n` cannot
        // hold with a non-zero acc — reaching the final row consumes its
        // boundary down-step and resets the accumulator.)
        if i < n && acc.iter().any(|&v| v != 0.0) {
            carries.push(CarryOut {
                row: i,
                partial: acc,
            });
        }
    }

    // Serial deterministic fix-up: fold every carry into its owning row.
    for carry in &carries {
        let base = carry.row * feat_dim;
        for (k, &v) in carry.partial.iter().enumerate() {
            y[base + k] += v;
        }
    }

    Ok(y)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Build a deterministic power-law-ish graph: a few hub nodes own most edges.
    fn skewed_graph(n: usize, hubs: usize, seed: u64) -> CsrGraph {
        let mut rng = LcgRng::new(seed);
        let mut edges: Vec<(usize, usize)> = Vec::new();
        // Hubs connect to a large fraction of the graph.
        for h in 0..hubs.min(n) {
            for d in 0..n {
                if d != h {
                    edges.push((h, d));
                }
            }
        }
        // Every other node gets a couple of random edges.
        for sidx in hubs..n {
            let deg = 1 + (rng.next_u32() as usize % 3);
            for _ in 0..deg {
                let d = rng.next_u32() as usize % n;
                if d != sidx {
                    edges.push((sidx, d));
                }
            }
        }
        CsrGraph::from_edges(n, &edges).expect("test invariant: value must be valid")
    }

    fn random_features(n: usize, feat_dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n * feat_dim)
            .map(|_| {
                let u = rng.next_u32() as f64 / 2f64.powi(32);
                (u as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn matches_naive_on_skewed_graph() {
        let g = skewed_graph(64, 3, 0xABCD);
        let feat_dim = 5;
        let x = random_features(g.n_nodes(), feat_dim, 0x1234);
        let cfg = BalancedSpmvConfig::new(8, 256).expect("config");
        let balanced = balanced_spmv(&g, &x, feat_dim, &cfg).expect("balanced");
        let naive = g.spmv(&x, feat_dim).expect("naive");
        assert_eq!(balanced.len(), naive.len());
        for (b, n_) in balanced.iter().zip(naive.iter()) {
            assert!((b - n_).abs() < 1e-4, "balanced {b} vs naive {n_}");
        }
    }

    #[test]
    fn matches_naive_across_segment_counts() {
        // Exercise many segmentations to stress the merge-path / carry logic.
        let g = skewed_graph(40, 2, 0x55AA);
        let feat_dim = 3;
        let x = random_features(g.n_nodes(), feat_dim, 0x9999);
        let naive = g.spmv(&x, feat_dim).expect("naive");
        for ips in [1usize, 2, 3, 5, 7, 16, 64, 4096] {
            let cfg = BalancedSpmvConfig::new(ips, 4096).expect("config");
            let balanced = balanced_spmv(&g, &x, feat_dim, &cfg).expect("balanced");
            for (b, n_) in balanced.iter().zip(naive.iter()) {
                assert!((b - n_).abs() < 1e-4, "ips={ips}: {b} vs {n_}");
            }
        }
    }

    #[test]
    fn single_processor_equals_naive() {
        let g = skewed_graph(32, 4, 0x0F0F);
        let feat_dim = 2;
        let x = random_features(g.n_nodes(), feat_dim, 0x4242);
        // One huge segment ⇒ pure serial walk, no carries across segments.
        let cfg = BalancedSpmvConfig::new(usize::MAX / 4, 1).expect("config");
        let balanced = balanced_spmv(&g, &x, feat_dim, &cfg).expect("balanced");
        let naive = g.spmv(&x, feat_dim).expect("naive");
        for (b, n_) in balanced.iter().zip(naive.iter()) {
            assert!((b - n_).abs() < 1e-5);
        }
    }

    #[test]
    fn dimension_mismatch_errors() {
        let g = CsrGraph::from_edges(3, &[(0, 1)]).expect("graph");
        let cfg = BalancedSpmvConfig::default();
        let err = balanced_spmv(&g, &[1.0, 2.0], 2, &cfg);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn empty_edges_yields_zero() {
        // No edges ⇒ result is all zeros (rows still consume down-steps).
        let g = CsrGraph::new(4, vec![0, 0, 0, 0, 0], vec![]).expect("graph");
        let x = vec![1.0_f32; 4 * 3];
        let cfg = BalancedSpmvConfig::new(2, 16).expect("config");
        let y = balanced_spmv(&g, &x, 3, &cfg).expect("balanced");
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn merge_path_search_endpoints() {
        // row_ptr for 3 rows with 0,2,1 nonzeros ⇒ [0,0,2,3], nnz=3.
        let row_ptr = vec![0usize, 0, 2, 3];
        let nnz = 3;
        // diag 0 ⇒ (0,0).
        assert_eq!(merge_path_search(0, &row_ptr, nnz, 3), 0);
        // Final anti-diagonal n+nnz=6 maps to i=n=3 (all rows finished).
        assert_eq!(merge_path_search(6, &row_ptr, nnz, 3), 3);
        // Anti-diagonals should be monotone non-decreasing in `i`.
        let mut prev = 0;
        for diag in 0..=6 {
            let i = merge_path_search(diag, &row_ptr, nnz, 3);
            assert!(i >= prev && i <= 3, "diag {diag}: i={i}");
            prev = i;
        }
    }

    #[test]
    fn config_rejects_zero_fields() {
        assert!(BalancedSpmvConfig::new(0, 4).is_err());
        assert!(BalancedSpmvConfig::new(4, 0).is_err());
    }

    #[test]
    fn segment_count_clamps() {
        assert_eq!(segment_count(0, 4, 8), 1);
        assert_eq!(segment_count(100, 10, 4), 4); // clamped by max_processors
        assert_eq!(segment_count(3, 1, 16), 3); // clamped by total_work
    }

    #[test]
    fn single_dense_row_balances() {
        // Star graph: node 0 connects to all others ⇒ extreme skew.
        let n = 100;
        let edges: Vec<(usize, usize)> = (1..n).map(|d| (0usize, d)).collect();
        let g = CsrGraph::from_edges(n, &edges).expect("graph");
        let feat_dim = 4;
        let x = random_features(n, feat_dim, 0x7777);
        // Fine segments force the dense row 0 to span many merge segments.
        let cfg = BalancedSpmvConfig::new(4, 4096).expect("config");
        let balanced = balanced_spmv(&g, &x, feat_dim, &cfg).expect("balanced");
        let naive = g.spmv(&x, feat_dim).expect("naive");
        for (b, n_) in balanced.iter().zip(naive.iter()) {
            assert!((b - n_).abs() < 1e-4, "{b} vs {n_}");
        }
    }
}
