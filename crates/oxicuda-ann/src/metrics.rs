//! Recall@K and recall–latency Pareto-frontier helpers.
//!
//! Tooling for evaluating an approximate index against exact ground truth and
//! for turning a parameter sweep (e.g. `nprobe` for IVF, `ef_search` for HNSW)
//! into a recall-vs-throughput trade-off curve.
//!
//! * [`recall_at_k`] — fraction of the true top-`k` neighbours that an
//!   approximate result set recovers, averaged over queries.
//! * [`ParetoPoint`] / [`pareto_frontier`] — given `(recall, qps)` measurements
//!   for several configurations, extract the non-dominated (Pareto-optimal)
//!   front: a point survives iff no other point has *both* higher recall and
//!   higher throughput.
//! * [`ParetoSweep`] — accumulate measurements and emit a header-prefixed CSV
//!   (`config,recall,qps,latency_us`) for external plotting — no plotting crate
//!   is pulled in, only a `String` is produced.
//!
//! Throughput (`qps`, queries-per-second) is derived from a measured per-query
//! latency: `qps = 1e6 / latency_us`. Latencies are supplied by the caller (this
//! module performs no timing itself, keeping it host-agnostic and deterministic
//! under test).

use crate::error::{AnnError, AnnResult};

/// Average recall@k of approximate results against exact ground truth.
///
/// * `approx[q]` is the approximate neighbour-id list for query `q` (only the
///   first `k` entries are considered; extra entries are ignored).
/// * `truth[q]` is the exact top-`k` neighbour-id list for query `q`.
///
/// Returns `hits / (n_queries * k)` where `hits` counts, per query, how many of
/// the true top-`k` ids appear among the approximate top-`k`.
///
/// # Errors
/// - [`AnnError::EmptyInput`] if there are no queries or `k == 0`.
/// - [`AnnError::DimensionMismatch`] if `approx.len() != truth.len()`.
pub fn recall_at_k(approx: &[Vec<u32>], truth: &[Vec<u32>], k: usize) -> AnnResult<f32> {
    if approx.is_empty() || k == 0 {
        return Err(AnnError::EmptyInput);
    }
    if approx.len() != truth.len() {
        return Err(AnnError::DimensionMismatch {
            expected: truth.len(),
            got: approx.len(),
        });
    }
    let mut hits = 0usize;
    let mut denom = 0usize;
    for (a, t) in approx.iter().zip(truth.iter()) {
        let kt = k.min(t.len());
        if kt == 0 {
            continue;
        }
        denom += kt;
        // Linear membership test over the (small) truncated approx list.
        let a_top: &[u32] = if a.len() > k { &a[..k] } else { a };
        for tid in t.iter().take(kt) {
            if a_top.contains(tid) {
                hits += 1;
            }
        }
    }
    if denom == 0 {
        return Err(AnnError::EmptyInput);
    }
    Ok(hits as f32 / denom as f32)
}

/// Compute exact top-`k` neighbour ids of `query` over a row-major `[n × dim]`
/// corpus, by squared L2 (ground-truth generator for [`recall_at_k`]).
///
/// # Errors
/// - [`AnnError::EmptyInput`] if `n == 0` or `k == 0`.
/// - [`AnnError::InvalidVectorDim`] if `dim == 0`.
/// - [`AnnError::DimensionMismatch`] if `data.len() != n * dim` or
///   `query.len() != dim`.
pub fn exact_topk_ids(
    data: &[f32],
    n: usize,
    dim: usize,
    query: &[f32],
    k: usize,
) -> AnnResult<Vec<u32>> {
    if n == 0 || k == 0 {
        return Err(AnnError::EmptyInput);
    }
    if dim == 0 {
        return Err(AnnError::InvalidVectorDim { dim });
    }
    if data.len() != n * dim {
        return Err(AnnError::DimensionMismatch {
            expected: n * dim,
            got: data.len(),
        });
    }
    if query.len() != dim {
        return Err(AnnError::DimensionMismatch {
            expected: dim,
            got: query.len(),
        });
    }
    let mut scored: Vec<(u32, f32)> = (0..n)
        .map(|i| {
            let v = &data[i * dim..(i + 1) * dim];
            let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
            (i as u32, d)
        })
        .collect();
    scored.sort_unstable_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    Ok(scored.into_iter().take(k).map(|(id, _)| id).collect())
}

/// A single measured operating point in a recall–throughput sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParetoPoint {
    /// Sweep parameter value (e.g. `nprobe` or `ef_search`).
    pub config: u32,
    /// Average recall@k in `[0, 1]`.
    pub recall: f32,
    /// Throughput in queries-per-second.
    pub qps: f32,
    /// Per-query latency in microseconds.
    pub latency_us: f32,
}

impl ParetoPoint {
    /// Build a point from a parameter value, recall, and per-query latency
    /// (microseconds). Derives `qps = 1e6 / latency_us`.
    #[must_use]
    pub fn from_latency(config: u32, recall: f32, latency_us: f32) -> Self {
        let qps = if latency_us > 0.0 {
            1.0e6 / latency_us
        } else {
            f32::INFINITY
        };
        Self {
            config,
            recall,
            qps,
            latency_us,
        }
    }

    /// `true` when `self` is dominated by `other`: `other` is at least as good on
    /// both recall and throughput and strictly better on at least one.
    #[must_use]
    pub fn dominated_by(&self, other: &ParetoPoint) -> bool {
        let ge = other.recall >= self.recall && other.qps >= self.qps;
        let gt = other.recall > self.recall || other.qps > self.qps;
        ge && gt
    }
}

/// Extract the Pareto-optimal frontier (maximising both recall and `qps`).
///
/// A point is kept iff no *other* point dominates it. Exact duplicates are
/// collapsed to a single representative. The returned front is sorted ascending
/// by recall (and, for ties, descending by `qps`).
#[must_use]
pub fn pareto_frontier(points: &[ParetoPoint]) -> Vec<ParetoPoint> {
    let mut front: Vec<ParetoPoint> = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let mut dominated = false;
        for (j, q) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            if p.dominated_by(q) {
                dominated = true;
                break;
            }
        }
        if dominated {
            continue;
        }
        // Skip if an identical (recall, qps) representative is already present.
        let dup = front.iter().any(|f| {
            (f.recall - p.recall).abs() < f32::EPSILON && (f.qps - p.qps).abs() < f32::EPSILON
        });
        if !dup {
            front.push(*p);
        }
    }
    front.sort_by(|a, b| {
        a.recall
            .partial_cmp(&b.recall)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.qps
                    .partial_cmp(&a.qps)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    front
}

/// Accumulator for a recall–latency parameter sweep with CSV export.
#[derive(Debug, Default, Clone)]
pub struct ParetoSweep {
    points: Vec<ParetoPoint>,
}

impl ParetoSweep {
    /// Create an empty sweep.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Record an operating point from a measured per-query latency (µs).
    pub fn record(&mut self, config: u32, recall: f32, latency_us: f32) {
        self.points
            .push(ParetoPoint::from_latency(config, recall, latency_us));
    }

    /// Push a fully-specified point.
    pub fn push(&mut self, point: ParetoPoint) {
        self.points.push(point);
    }

    /// All recorded points (insertion order).
    #[must_use]
    pub fn points(&self) -> &[ParetoPoint] {
        &self.points
    }

    /// Number of recorded points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// `true` when nothing has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The Pareto-optimal subset of the recorded points.
    #[must_use]
    pub fn frontier(&self) -> Vec<ParetoPoint> {
        pareto_frontier(&self.points)
    }

    /// Render all recorded points as CSV with a header line. Columns:
    /// `config,recall,qps,latency_us`. Rows follow insertion order.
    #[must_use]
    pub fn to_csv(&self) -> String {
        self.csv_from(&self.points)
    }

    /// Render only the Pareto frontier as CSV (same columns / header).
    #[must_use]
    pub fn frontier_to_csv(&self) -> String {
        let f = self.frontier();
        self.csv_from(&f)
    }

    fn csv_from(&self, rows: &[ParetoPoint]) -> String {
        let mut s = String::from("config,recall,qps,latency_us\n");
        for p in rows {
            s.push_str(&format!(
                "{},{:.6},{:.6},{:.6}\n",
                p.config, p.recall, p.qps, p.latency_us
            ));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── recall@k ───────────────────────────────────────────────────────────

    #[test]
    fn recall_perfect() {
        let approx = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let truth = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let r = recall_at_k(&approx, &truth, 3).expect("recall");
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_half() {
        // Each query gets 1 of 2 true neighbours right.
        let approx = vec![vec![1, 9], vec![4, 9]];
        let truth = vec![vec![1, 2], vec![4, 5]];
        let r = recall_at_k(&approx, &truth, 2).expect("recall");
        assert!((r - 0.5).abs() < 1e-6, "r={r}");
    }

    #[test]
    fn recall_order_independent() {
        // Recall is a set measure; order in approx must not matter.
        let approx = vec![vec![3, 2, 1]];
        let truth = vec![vec![1, 2, 3]];
        let r = recall_at_k(&approx, &truth, 3).expect("recall");
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_truncates_to_k() {
        // Extra approx entries beyond k must be ignored.
        let approx = vec![vec![9, 9, 1]]; // k=2 -> only [9,9] counts
        let truth = vec![vec![1, 2]];
        let r = recall_at_k(&approx, &truth, 2).expect("recall");
        assert!((r - 0.0).abs() < 1e-6, "r={r}");
    }

    #[test]
    fn recall_empty_errors() {
        let empty: Vec<Vec<u32>> = Vec::new();
        assert!(recall_at_k(&empty, &empty, 3).is_err());
        assert!(recall_at_k(&[vec![1]], &[vec![1]], 0).is_err());
    }

    #[test]
    fn recall_length_mismatch_errors() {
        assert!(recall_at_k(&[vec![1]], &[vec![1], vec![2]], 1).is_err());
    }

    // ── exact ground truth ─────────────────────────────────────────────────

    #[test]
    fn exact_topk_finds_self() {
        let mut rng = LcgRng::new(1);
        let n = 50;
        let dim = 6;
        let data: Vec<f32> = (0..n * dim).map(|_| rng.next_f32()).collect();
        for i in 0..n {
            let q = &data[i * dim..(i + 1) * dim];
            let ids = exact_topk_ids(&data, n, dim, q, 1).expect("topk");
            assert_eq!(ids[0] as usize, i);
        }
    }

    #[test]
    fn exact_topk_validates() {
        assert!(exact_topk_ids(&[], 0, 4, &[0.0; 4], 1).is_err());
        assert!(exact_topk_ids(&[0.0; 4], 1, 0, &[], 1).is_err());
        assert!(exact_topk_ids(&[0.0; 3], 1, 4, &[0.0; 4], 1).is_err());
        assert!(exact_topk_ids(&[0.0; 4], 1, 4, &[0.0; 3], 1).is_err());
    }

    #[test]
    fn recall_against_exact_is_one_for_exact_results() {
        // An "approximate" search that is actually exact must yield recall 1.0.
        let mut rng = LcgRng::new(2);
        let n = 40;
        let dim = 5;
        let k = 5;
        let data: Vec<f32> = (0..n * dim).map(|_| rng.next_f32()).collect();
        let queries: Vec<f32> = (0..10 * dim).map(|_| rng.next_f32()).collect();
        let mut approx = Vec::new();
        let mut truth = Vec::new();
        for qi in 0..10 {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let gt = exact_topk_ids(&data, n, dim, q, k).expect("gt");
            approx.push(gt.clone());
            truth.push(gt);
        }
        let r = recall_at_k(&approx, &truth, k).expect("recall");
        assert!((r - 1.0).abs() < 1e-6);
    }

    // ── Pareto ─────────────────────────────────────────────────────────────

    #[test]
    fn pareto_point_domination() {
        let a = ParetoPoint {
            config: 1,
            recall: 0.8,
            qps: 1000.0,
            latency_us: 1000.0,
        };
        let b = ParetoPoint {
            config: 2,
            recall: 0.9,
            qps: 1200.0,
            latency_us: 833.0,
        };
        assert!(a.dominated_by(&b));
        assert!(!b.dominated_by(&a));
        // Equal points do not dominate each other.
        assert!(!a.dominated_by(&a));
    }

    #[test]
    fn pareto_frontier_typical_tradeoff() {
        // Classic ANN sweep: higher config → higher recall but lower qps.
        // All four points are mutually non-dominated → all on the front.
        let pts = vec![
            ParetoPoint::from_latency(1, 0.70, 200.0),  // qps 5000
            ParetoPoint::from_latency(2, 0.85, 400.0),  // qps 2500
            ParetoPoint::from_latency(4, 0.93, 800.0),  // qps 1250
            ParetoPoint::from_latency(8, 0.98, 1600.0), // qps 625
        ];
        let front = pareto_frontier(&pts);
        assert_eq!(front.len(), 4);
        // Sorted ascending by recall.
        for w in front.windows(2) {
            assert!(w[0].recall <= w[1].recall);
        }
    }

    #[test]
    fn pareto_frontier_drops_dominated() {
        let pts = vec![
            ParetoPoint::from_latency(1, 0.70, 200.0), // qps 5000
            ParetoPoint::from_latency(2, 0.60, 400.0), // dominated (worse recall & qps)
            ParetoPoint::from_latency(4, 0.95, 800.0), // qps 1250
        ];
        let front = pareto_frontier(&pts);
        assert_eq!(front.len(), 2);
        assert!(front.iter().all(|p| p.config != 2));
    }

    #[test]
    fn pareto_frontier_collapses_duplicates() {
        let p = ParetoPoint::from_latency(1, 0.9, 500.0);
        let q = ParetoPoint::from_latency(2, 0.9, 500.0);
        let front = pareto_frontier(&[p, q]);
        assert_eq!(front.len(), 1);
    }

    #[test]
    fn pareto_frontier_empty() {
        assert!(pareto_frontier(&[]).is_empty());
    }

    // ── sweep + CSV ────────────────────────────────────────────────────────

    #[test]
    fn sweep_record_and_csv() {
        let mut sweep = ParetoSweep::new();
        assert!(sweep.is_empty());
        sweep.record(1, 0.70, 200.0);
        sweep.record(2, 0.85, 400.0);
        sweep.record(4, 0.95, 800.0);
        assert_eq!(sweep.len(), 3);

        let csv = sweep.to_csv();
        assert!(csv.starts_with("config,recall,qps,latency_us\n"));
        // One header line + three data lines (+ trailing newline → 4 splits).
        assert_eq!(csv.lines().count(), 4);
        assert!(csv.contains("\n1,0.7000"));

        let front = sweep.frontier();
        assert_eq!(front.len(), 3);
        let fcsv = sweep.frontier_to_csv();
        assert!(fcsv.starts_with("config,recall,qps,latency_us\n"));
    }

    #[test]
    fn sweep_qps_derivation() {
        let mut sweep = ParetoSweep::new();
        sweep.record(1, 0.9, 1000.0); // 1 ms → 1000 qps
        let p = sweep.points()[0];
        assert!((p.qps - 1000.0).abs() < 1e-3, "qps={}", p.qps);
    }
}
