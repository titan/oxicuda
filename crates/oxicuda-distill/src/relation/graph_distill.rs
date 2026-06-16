//! Relational graph distillation with edge-feature aggregation.
//!
//! Knowledge Distillation via Instance Relationship Graph (Liu et al. 2019 CVPR).
//!
//! Each mini-batch of embeddings is viewed as an **instance relationship graph**: nodes are
//! sample embeddings and edges encode pairwise relations (Euclidean distance or cosine
//! similarity). The student is trained to match the teacher's **edge features** (the
//! off-diagonal relation matrix) *and* the **vertex-aggregated features** (per-node summaries
//! of incident edges). This transfers the geometry of the teacher's representation space
//! rather than individual activations, and is fully deterministic (no RNG).

use crate::error::{DistillError, DistillResult};

const EPS: f32 = 1e-8;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration enums
// ─────────────────────────────────────────────────────────────────────────────

/// Pairwise relation metric used to build the instance relationship graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationMetric {
    /// L2 (Euclidean) distance between embeddings; always `≥ 0`.
    Euclidean,
    /// Cosine similarity between embeddings; lies in `[-1, 1]` (scale-invariant).
    Cosine,
}

/// Vertex aggregation scheme over each node's incident (off-diagonal) edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    /// Average of the off-diagonal row: `h_i = (Σ_{j≠i} A_ij) / (n − 1)`.
    Mean,
    /// Row sum normalized by `n − 1` (identical to [`Aggregation::Mean`] for a full graph,
    /// but kept distinct to express "summed then normalized" semantics).
    SumNormalized,
}

/// Configuration for relational graph distillation.
#[derive(Debug, Clone)]
pub struct GraphDistillConfig {
    /// Pairwise relation metric.
    pub metric: RelationMetric,
    /// Vertex aggregation scheme.
    pub aggregation: Aggregation,
    /// Weight on the vertex (node-feature) loss term.
    pub lambda_vertex: f32,
    /// Weight on the edge (relation-matrix) loss term.
    pub lambda_edge: f32,
}

impl Default for GraphDistillConfig {
    fn default() -> Self {
        Self {
            metric: RelationMetric::Euclidean,
            aggregation: Aggregation::Mean,
            lambda_vertex: 1.0,
            lambda_edge: 1.0,
        }
    }
}

impl GraphDistillConfig {
    /// Validate the loss weights.
    fn validate(&self) -> DistillResult<()> {
        if self.lambda_vertex < 0.0 || self.lambda_edge < 0.0 {
            return Err(DistillError::InvalidConfig {
                msg: format!(
                    "GraphDistillConfig: lambda_vertex ({}) and lambda_edge ({}) must be >= 0",
                    self.lambda_vertex, self.lambda_edge
                ),
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Relation primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean (L2) distance between two equal-length embedding rows.
fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Cosine similarity between two embedding rows, guarding against zero norms.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + EPS)
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph construction & aggregation
// ─────────────────────────────────────────────────────────────────────────────

/// Build the `n × n` instance relationship graph (flat row-major).
///
/// `embeddings` is `[n · dim]` row-major. `A_ij = relation(f_i, f_j)` with the diagonal
/// forced to `0`. For [`RelationMetric::Euclidean`] entries are L2 distances; for
/// [`RelationMetric::Cosine`] entries are cosine similarities (zero-norm guarded).
pub fn build_relation_graph(
    embeddings: &[f32],
    n: usize,
    dim: usize,
    metric: RelationMetric,
) -> DistillResult<Vec<f32>> {
    if embeddings.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if n < 2 {
        return Err(DistillError::InvalidConfig {
            msg: format!("build_relation_graph: need at least 2 nodes, got n={n}"),
        });
    }
    if dim == 0 {
        return Err(DistillError::InvalidConfig {
            msg: "build_relation_graph: dim must be > 0".into(),
        });
    }
    let expected = n * dim;
    if embeddings.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: embeddings.len(),
        });
    }

    let mut graph = vec![0.0_f32; n * n];
    for i in 0..n {
        let row_i =
            embeddings
                .get(i * dim..(i + 1) * dim)
                .ok_or_else(|| DistillError::Internal {
                    msg: "build_relation_graph: row i out of bounds".into(),
                })?;
        for j in (i + 1)..n {
            let row_j =
                embeddings
                    .get(j * dim..(j + 1) * dim)
                    .ok_or_else(|| DistillError::Internal {
                        msg: "build_relation_graph: row j out of bounds".into(),
                    })?;
            let rel = match metric {
                RelationMetric::Euclidean => euclidean(row_i, row_j),
                RelationMetric::Cosine => cosine(row_i, row_j),
            };
            // Symmetric: write both (i,j) and (j,i); diagonal stays 0.
            let ij = graph
                .get_mut(i * n + j)
                .ok_or_else(|| DistillError::Internal {
                    msg: "build_relation_graph: (i,j) out of bounds".into(),
                })?;
            *ij = rel;
            let ji = graph
                .get_mut(j * n + i)
                .ok_or_else(|| DistillError::Internal {
                    msg: "build_relation_graph: (j,i) out of bounds".into(),
                })?;
            *ji = rel;
        }
    }
    Ok(graph)
}

/// Aggregate vertex features from the relation graph.
///
/// `graph` is the `n × n` flat adjacency. For node `i`, `h_i` aggregates over `j ≠ i`:
/// * [`Aggregation::Mean`] — average of the off-diagonal row.
/// * [`Aggregation::SumNormalized`] — row sum divided by `n − 1`.
///
/// Returns a length-`n` vector. With `n < 2` (no off-diagonal edges) every entry is `0`.
#[must_use]
pub fn aggregate_vertices(graph: &[f32], n: usize, agg: Aggregation) -> Vec<f32> {
    let mut h = vec![0.0_f32; n];
    if n < 2 {
        return h;
    }
    let denom = (n - 1) as f32;
    for i in 0..n {
        let mut sum = 0.0_f32;
        for j in 0..n {
            if i == j {
                continue;
            }
            if let Some(&v) = graph.get(i * n + j) {
                sum += v;
            }
        }
        let value = match agg {
            Aggregation::Mean | Aggregation::SumNormalized => sum / denom,
        };
        if let Some(slot) = h.get_mut(i) {
            *slot = value;
        }
    }
    h
}

/// Mean-squared error over the **off-diagonal** entries of two `n × n` graphs.
fn off_diagonal_mse(a: &[f32], b: &[f32], n: usize) -> DistillResult<f32> {
    if n < 2 {
        return Ok(0.0);
    }
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let av = *a.get(i * n + j).ok_or_else(|| DistillError::Internal {
                msg: "off_diagonal_mse: a index out of bounds".into(),
            })?;
            let bv = *b.get(i * n + j).ok_or_else(|| DistillError::Internal {
                msg: "off_diagonal_mse: b index out of bounds".into(),
            })?;
            sum += (av - bv) * (av - bv);
            count += 1;
        }
    }
    if count == 0 {
        return Ok(0.0);
    }
    Ok(sum / count as f32)
}

/// Mean-squared error over two equal-length vectors.
fn vector_mse(a: &[f32], b: &[f32]) -> DistillResult<f32> {
    if a.len() != b.len() {
        return Err(DistillError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Ok(0.0);
    }
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    Ok(sum / a.len() as f32)
}

/// Compute the relational graph distillation loss.
///
/// `loss = lambda_edge · MSE(off-diag A_student, off-diag A_teacher)`
/// `     + lambda_vertex · MSE(h_student, h_teacher)`.
///
/// Both embedding sets are `[n · dim]` row-major. Deterministic; uses no RNG.
pub fn graph_distill_loss(
    student_emb: &[f32],
    teacher_emb: &[f32],
    n: usize,
    dim: usize,
    cfg: &GraphDistillConfig,
) -> DistillResult<f32> {
    cfg.validate()?;
    if student_emb.is_empty() || teacher_emb.is_empty() {
        return Err(DistillError::EmptyInput);
    }
    if n < 2 {
        return Err(DistillError::InvalidConfig {
            msg: format!("graph_distill_loss: need at least 2 nodes, got n={n}"),
        });
    }
    if dim == 0 {
        return Err(DistillError::InvalidConfig {
            msg: "graph_distill_loss: dim must be > 0".into(),
        });
    }
    let expected = n * dim;
    if student_emb.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: student_emb.len(),
        });
    }
    if teacher_emb.len() != expected {
        return Err(DistillError::DimensionMismatch {
            expected,
            got: teacher_emb.len(),
        });
    }

    let g_student = build_relation_graph(student_emb, n, dim, cfg.metric)?;
    let g_teacher = build_relation_graph(teacher_emb, n, dim, cfg.metric)?;

    let edge_loss = off_diagonal_mse(&g_student, &g_teacher, n)?;

    let h_student = aggregate_vertices(&g_student, n, cfg.aggregation);
    let h_teacher = aggregate_vertices(&g_teacher, n, cfg.aggregation);
    let vertex_loss = vector_mse(&h_student, &h_teacher)?;

    Ok(cfg.lambda_edge * edge_loss + cfg.lambda_vertex * vertex_loss)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(metric: RelationMetric, agg: Aggregation) -> GraphDistillConfig {
        GraphDistillConfig {
            metric,
            aggregation: agg,
            lambda_vertex: 1.0,
            lambda_edge: 1.0,
        }
    }

    // ── 1. graph symmetric (euclidean) ──────────────────────────────────────
    #[test]
    fn graph_symmetric_euclidean() {
        let n = 4usize;
        let dim = 3usize;
        let emb: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let g = build_relation_graph(&emb, n, dim, RelationMetric::Euclidean)
            .expect("build_relation_graph should succeed");
        for i in 0..n {
            for j in 0..n {
                let gij = g[i * n + j];
                let gji = g[j * n + i];
                assert!(
                    (gij - gji).abs() < 1e-6,
                    "A[{i},{j}]={gij} != A[{j},{i}]={gji}"
                );
            }
        }
    }

    // ── 2. graph symmetric (cosine) ─────────────────────────────────────────
    #[test]
    fn graph_symmetric_cosine() {
        let n = 5usize;
        let dim = 4usize;
        let emb: Vec<f32> = (0..n * dim).map(|i| (i as f32 + 1.0) * 0.2).collect();
        let g = build_relation_graph(&emb, n, dim, RelationMetric::Cosine)
            .expect("build_relation_graph should succeed");
        for i in 0..n {
            for j in 0..n {
                assert!((g[i * n + j] - g[j * n + i]).abs() < 1e-6);
            }
        }
    }

    // ── 3. diagonal is zero ─────────────────────────────────────────────────
    #[test]
    fn diagonal_zero() {
        let n = 4usize;
        let dim = 2usize;
        let emb: Vec<f32> = (0..n * dim).map(|i| i as f32 + 1.0).collect();
        for metric in [RelationMetric::Euclidean, RelationMetric::Cosine] {
            let g = build_relation_graph(&emb, n, dim, metric)
                .expect("build_relation_graph should succeed");
            for i in 0..n {
                assert_eq!(
                    g[i * n + i],
                    0.0,
                    "diagonal must be 0 for metric {metric:?}"
                );
            }
        }
    }

    // ── 4. identical student/teacher → loss 0 ───────────────────────────────
    #[test]
    fn identical_zero_loss() {
        let n = 4usize;
        let dim = 3usize;
        let emb: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.5).collect();
        for metric in [RelationMetric::Euclidean, RelationMetric::Cosine] {
            let c = cfg(metric, Aggregation::Mean);
            let loss = graph_distill_loss(&emb, &emb, n, dim, &c)
                .expect("graph_distill_loss should succeed");
            assert!(loss.abs() < 1e-6, "identical → 0, got {loss} ({metric:?})");
        }
    }

    // ── 5. cosine values within [-1, 1] ─────────────────────────────────────
    #[test]
    fn cosine_in_range() {
        let n = 6usize;
        let dim = 4usize;
        let emb: Vec<f32> = (0..n * dim).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
        let g = build_relation_graph(&emb, n, dim, RelationMetric::Cosine)
            .expect("build_relation_graph should succeed");
        for &v in &g {
            assert!(
                (-1.0 - 1e-5..=1.0 + 1e-5).contains(&v),
                "cosine out of range: {v}"
            );
        }
    }

    // ── 6. euclidean values are non-negative ────────────────────────────────
    #[test]
    fn euclidean_nonneg() {
        let n = 5usize;
        let dim = 3usize;
        let emb: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.7 - 3.0).collect();
        let g = build_relation_graph(&emb, n, dim, RelationMetric::Euclidean)
            .expect("build_relation_graph should succeed");
        for &v in &g {
            assert!(v >= 0.0, "euclidean distance must be >= 0, got {v}");
        }
    }

    // ── 7. cosine is scale-invariant; euclidean is not ──────────────────────
    #[test]
    fn cosine_scale_invariant_euclidean_not() {
        let n = 4usize;
        let dim = 3usize;
        let emb: Vec<f32> = (0..n * dim).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let scaled: Vec<f32> = emb.iter().map(|&v| v * 5.0).collect();

        let cos1 = build_relation_graph(&emb, n, dim, RelationMetric::Cosine)
            .expect("build_relation_graph should succeed");
        let cos2 = build_relation_graph(&scaled, n, dim, RelationMetric::Cosine)
            .expect("build_relation_graph should succeed");
        for (a, b) in cos1.iter().zip(cos2.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "cosine must be scale-invariant: {a} vs {b}"
            );
        }

        let euc1 = build_relation_graph(&emb, n, dim, RelationMetric::Euclidean)
            .expect("build_relation_graph should succeed");
        let euc2 = build_relation_graph(&scaled, n, dim, RelationMetric::Euclidean)
            .expect("build_relation_graph should succeed");
        let changed = euc1
            .iter()
            .zip(euc2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-3);
        assert!(changed, "euclidean must change under scaling");
    }

    // ── 8. vertex Mean correct by hand on a 3-node graph ────────────────────
    #[test]
    fn vertex_mean_by_hand() {
        // Build a tiny 1-D embedding so distances are easy: f = [0], [1], [3].
        let emb = vec![0.0_f32, 1.0, 3.0];
        let n = 3usize;
        let dim = 1usize;
        let g = build_relation_graph(&emb, n, dim, RelationMetric::Euclidean)
            .expect("build_relation_graph should succeed");
        // A = [[0,1,3],[1,0,2],[3,2,0]]
        // h0 = (1+3)/2 = 2 ; h1 = (1+2)/2 = 1.5 ; h2 = (3+2)/2 = 2.5
        let h = aggregate_vertices(&g, n, Aggregation::Mean);
        assert!((h[0] - 2.0).abs() < 1e-6, "h0={}", h[0]);
        assert!((h[1] - 1.5).abs() < 1e-6, "h1={}", h[1]);
        assert!((h[2] - 2.5).abs() < 1e-6, "h2={}", h[2]);
    }

    // ── 9. SumNormalized matches Mean for a full graph ──────────────────────
    #[test]
    fn sum_normalized_matches_mean() {
        let emb = vec![0.0_f32, 1.0, 3.0, 6.0];
        let n = 4usize;
        let dim = 1usize;
        let g = build_relation_graph(&emb, n, dim, RelationMetric::Euclidean)
            .expect("build_relation_graph should succeed");
        let hm = aggregate_vertices(&g, n, Aggregation::Mean);
        let hs = aggregate_vertices(&g, n, Aggregation::SumNormalized);
        for (a, b) in hm.iter().zip(hs.iter()) {
            assert!((a - b).abs() < 1e-6, "mean {a} != sum_norm {b}");
        }
    }

    // ── 10. edge loss == off-diagonal MSE by hand ───────────────────────────
    #[test]
    fn edge_loss_by_hand() {
        // 1-D embeddings: student f=[0,1,3], teacher f=[0,2,3].
        let s = vec![0.0_f32, 1.0, 3.0];
        let t = vec![0.0_f32, 2.0, 3.0];
        let n = 3usize;
        let dim = 1usize;
        // student A_off: (0,1)=1,(0,2)=3,(1,2)=2 (and symmetric).
        // teacher A_off: (0,1)=2,(0,2)=3,(1,2)=1.
        // diffs² per ordered off-diag pair: (1-2)²=1 ×2, (3-3)²=0 ×2, (2-1)²=1 ×2.
        // MSE over 6 ordered off-diag entries = (1+1+0+0+1+1)/6 = 4/6 = 0.6667.
        let c = GraphDistillConfig {
            metric: RelationMetric::Euclidean,
            aggregation: Aggregation::Mean,
            lambda_vertex: 0.0, // isolate the edge term
            lambda_edge: 1.0,
        };
        let loss =
            graph_distill_loss(&s, &t, n, dim, &c).expect("graph_distill_loss should succeed");
        assert!(
            (loss - 4.0 / 6.0).abs() < 1e-5,
            "edge loss={loss}, expected {}",
            4.0 / 6.0
        );
    }

    // ── 11. lambda_edge scales the loss ─────────────────────────────────────
    #[test]
    fn lambda_edge_scales() {
        let s = vec![0.0_f32, 1.0, 3.0];
        let t = vec![0.0_f32, 2.0, 3.0];
        let n = 3usize;
        let dim = 1usize;
        let mut c1 = cfg(RelationMetric::Euclidean, Aggregation::Mean);
        c1.lambda_vertex = 0.0;
        c1.lambda_edge = 1.0;
        let mut c2 = c1.clone();
        c2.lambda_edge = 4.0;
        let l1 =
            graph_distill_loss(&s, &t, n, dim, &c1).expect("graph_distill_loss should succeed");
        let l2 =
            graph_distill_loss(&s, &t, n, dim, &c2).expect("graph_distill_loss should succeed");
        assert!(
            (l2 - 4.0 * l1).abs() < 1e-5,
            "lambda_edge must scale: {l1} {l2}"
        );
    }

    // ── 12. lambda_vertex scales the loss ───────────────────────────────────
    #[test]
    fn lambda_vertex_scales() {
        let s = vec![0.0_f32, 1.0, 3.0];
        let t = vec![0.0_f32, 2.0, 3.0];
        let n = 3usize;
        let dim = 1usize;
        let mut c1 = cfg(RelationMetric::Euclidean, Aggregation::Mean);
        c1.lambda_edge = 0.0;
        c1.lambda_vertex = 1.0;
        let mut c2 = c1.clone();
        c2.lambda_vertex = 3.0;
        let l1 =
            graph_distill_loss(&s, &t, n, dim, &c1).expect("graph_distill_loss should succeed");
        let l2 =
            graph_distill_loss(&s, &t, n, dim, &c2).expect("graph_distill_loss should succeed");
        assert!(l1 > 0.0, "vertex loss should be positive here, got {l1}");
        assert!(
            (l2 - 3.0 * l1).abs() < 1e-5,
            "lambda_vertex must scale: {l1} {l2}"
        );
    }

    // ── 13. output sizes: graph n², h n ─────────────────────────────────────
    #[test]
    fn output_sizes() {
        let n = 5usize;
        let dim = 3usize;
        let emb: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let g = build_relation_graph(&emb, n, dim, RelationMetric::Euclidean)
            .expect("build_relation_graph should succeed");
        assert_eq!(g.len(), n * n, "graph must be n²");
        let h = aggregate_vertices(&g, n, Aggregation::Mean);
        assert_eq!(h.len(), n, "vertex features must be length n");
    }

    // ── 14. relabeling samples permutes the graph consistently ──────────────
    #[test]
    fn relabel_permutes_graph() {
        // Original order f0,f1,f2; permuted order f2,f0,f1 (perm: new k -> old p[k]).
        let dim = 2usize;
        let f0 = [1.0_f32, 0.0];
        let f1 = [0.0_f32, 2.0];
        let f2 = [3.0_f32, 1.0];
        let orig: Vec<f32> = [f0, f1, f2].concat();
        let perm = [2usize, 0, 1];
        let permuted: Vec<f32> = perm
            .iter()
            .flat_map(|&p| [f0, f1, f2][p].to_vec())
            .collect();
        let n = 3usize;
        let g = build_relation_graph(&orig, n, dim, RelationMetric::Euclidean)
            .expect("build_relation_graph should succeed");
        let gp = build_relation_graph(&permuted, n, dim, RelationMetric::Euclidean)
            .expect("build_relation_graph should succeed");
        // gp[a,b] must equal g[perm[a], perm[b]].
        for a in 0..n {
            for b in 0..n {
                let lhs = gp[a * n + b];
                let rhs = g[perm[a] * n + perm[b]];
                assert!(
                    (lhs - rhs).abs() < 1e-6,
                    "perm mismatch at ({a},{b}): {lhs} vs {rhs}"
                );
            }
        }
    }

    // ── 15. err: n < 2 ──────────────────────────────────────────────────────
    #[test]
    fn err_n_too_small() {
        let emb = vec![1.0_f32, 2.0, 3.0];
        let r = build_relation_graph(&emb, 1, 3, RelationMetric::Euclidean);
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
        let c = cfg(RelationMetric::Euclidean, Aggregation::Mean);
        let r2 = graph_distill_loss(&emb, &emb, 1, 3, &c);
        assert!(matches!(r2, Err(DistillError::InvalidConfig { .. })));
    }

    // ── 16. err: dim = 0 ────────────────────────────────────────────────────
    #[test]
    fn err_dim_zero() {
        let emb = vec![1.0_f32, 2.0];
        let r = build_relation_graph(&emb, 2, 0, RelationMetric::Euclidean);
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
    }

    // ── 17. err: student/teacher dim mismatch in loss ───────────────────────
    #[test]
    fn err_dim_mismatch() {
        let s = vec![0.0_f32; 3 * 4]; // n=3, dim=4
        let t = vec![0.0_f32; 3 * 4 - 2]; // wrong length
        let c = cfg(RelationMetric::Euclidean, Aggregation::Mean);
        let r = graph_distill_loss(&s, &t, 3, 4, &c);
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
    }

    // ── 18. err: n/dim mismatch with embedding length ───────────────────────
    #[test]
    fn err_embedding_length_mismatch() {
        let emb = vec![0.0_f32; 10]; // not 3*4=12
        let r = build_relation_graph(&emb, 3, 4, RelationMetric::Euclidean);
        assert!(matches!(r, Err(DistillError::DimensionMismatch { .. })));
    }

    // ── 19. err: empty input ────────────────────────────────────────────────
    #[test]
    fn err_empty_input() {
        let r = build_relation_graph(&[], 3, 4, RelationMetric::Cosine);
        assert!(matches!(r, Err(DistillError::EmptyInput)));
        let c = cfg(RelationMetric::Cosine, Aggregation::Mean);
        let r2 = graph_distill_loss(&[], &[], 3, 4, &c);
        assert!(matches!(r2, Err(DistillError::EmptyInput)));
    }

    // ── 20. err: negative lambda ────────────────────────────────────────────
    #[test]
    fn err_negative_lambda() {
        let emb = vec![1.0_f32; 3 * 2];
        let mut c = cfg(RelationMetric::Euclidean, Aggregation::Mean);
        c.lambda_vertex = -1.0;
        let r = graph_distill_loss(&emb, &emb, 3, 2, &c);
        assert!(matches!(r, Err(DistillError::InvalidConfig { .. })));
        let mut c2 = cfg(RelationMetric::Euclidean, Aggregation::Mean);
        c2.lambda_edge = -0.5;
        let r2 = graph_distill_loss(&emb, &emb, 3, 2, &c2);
        assert!(matches!(r2, Err(DistillError::InvalidConfig { .. })));
    }

    // ── 21. cosine zero-norm guard (no NaN) ─────────────────────────────────
    #[test]
    fn cosine_zero_norm_no_nan() {
        // One node is the zero vector → cosine with it must be finite (guarded).
        let emb = vec![0.0_f32, 0.0, 1.0, 2.0, 3.0, 4.0]; // n=3, dim=2; node0 = zero
        let g = build_relation_graph(&emb, 3, 2, RelationMetric::Cosine)
            .expect("build_relation_graph should succeed");
        for &v in &g {
            assert!(
                v.is_finite(),
                "cosine with zero-norm node must be finite, got {v}"
            );
        }
    }

    // ── 22. full loss combines edge + vertex terms ──────────────────────────
    #[test]
    fn full_loss_combines_terms() {
        let s = vec![0.0_f32, 1.0, 3.0];
        let t = vec![0.0_f32, 2.0, 3.0];
        let n = 3usize;
        let dim = 1usize;
        let mut edge_only = cfg(RelationMetric::Euclidean, Aggregation::Mean);
        edge_only.lambda_vertex = 0.0;
        let mut vertex_only = cfg(RelationMetric::Euclidean, Aggregation::Mean);
        vertex_only.lambda_edge = 0.0;
        let both = cfg(RelationMetric::Euclidean, Aggregation::Mean);
        let le = graph_distill_loss(&s, &t, n, dim, &edge_only)
            .expect("graph_distill_loss should succeed");
        let lv = graph_distill_loss(&s, &t, n, dim, &vertex_only)
            .expect("graph_distill_loss should succeed");
        let lb =
            graph_distill_loss(&s, &t, n, dim, &both).expect("graph_distill_loss should succeed");
        assert!(
            (lb - (le + lv)).abs() < 1e-5,
            "combined {lb} != edge {le} + vertex {lv}"
        );
    }
}
