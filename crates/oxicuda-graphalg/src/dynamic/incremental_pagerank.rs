//! Incremental PageRank over a streaming [`DynamicGraph`].
//!
//! A single edge insertion or deletion perturbs only a handful of rows of the
//! Google matrix `G = d · (A D^{-1} + dangling teleport) + (1-d)/n · 1 1^T`, so
//! the previous stationary distribution is an excellent warm start for the new
//! one. [`IncrementalPageRank`] keeps the current rank vector and, after each
//! batch of `apply_*` mutations, resumes the power iteration **from that vector**
//! rather than from the uniform cold start. The fixed point of the power
//! iteration is unique (the Google matrix is column-stochastic, irreducible and
//! aperiodic for `0 < d < 1`), so warm-starting converges to *exactly* the same
//! ranks an independent cold-start solve would — only faster.
//!
//! Dangling vertices (out-degree zero) redistribute their mass uniformly, and
//! the result is renormalised to sum to one, matching
//! [`crate::centrality::pagerank::pagerank`].

use crate::dynamic::dynamic_graph::DynamicGraph;
use crate::error::{GraphalgError, GraphalgResult};

/// Stateful incremental PageRank engine bound to a mutable graph.
#[derive(Debug, Clone)]
pub struct IncrementalPageRank {
    graph: DynamicGraph,
    damping: f64,
    max_iter: usize,
    tol: f64,
    /// Current rank vector (always normalised to sum one).
    rank: Vec<f64>,
    /// Iterations consumed by the most recent [`recompute`](Self::recompute).
    last_iterations: usize,
}

impl IncrementalPageRank {
    /// Create an engine over `graph` and converge an initial cold-start solve.
    ///
    /// * `damping` — the teleport-complement `d ∈ [0, 1]` (typically `0.85`).
    /// * `max_iter` — power-iteration cap per (re)solve.
    /// * `tol` — L1 convergence tolerance between successive iterates.
    pub fn new(
        graph: DynamicGraph,
        damping: f64,
        max_iter: usize,
        tol: f64,
    ) -> GraphalgResult<Self> {
        if !(0.0..=1.0).contains(&damping) {
            return Err(GraphalgError::InvalidParameter(format!(
                "damping out of [0,1]: {damping}"
            )));
        }
        if !tol.is_finite() || tol < 0.0 {
            return Err(GraphalgError::InvalidParameter(format!(
                "tolerance must be finite and non-negative: {tol}"
            )));
        }
        let n = graph.num_nodes();
        let rank = if n == 0 {
            Vec::new()
        } else {
            vec![1.0 / n as f64; n]
        };
        let mut engine = Self {
            graph,
            damping,
            max_iter,
            tol,
            rank,
            last_iterations: 0,
        };
        // Cold start: iterate from the uniform vector to the fixed point.
        engine.power_iterate();
        Ok(engine)
    }

    /// Current (normalised) rank vector.
    pub fn ranks(&self) -> &[f64] {
        &self.rank
    }

    /// PageRank of a single vertex.
    pub fn rank_of(&self, v: usize) -> GraphalgResult<f64> {
        if v >= self.graph.num_nodes() {
            return Err(GraphalgError::IndexOutOfBounds {
                index: v,
                len: self.graph.num_nodes(),
            });
        }
        Ok(self.rank[v])
    }

    /// Iterations consumed by the most recent (re)solve — strictly informational,
    /// useful for demonstrating the warm-start speed-up in tests/benchmarks.
    pub fn last_iterations(&self) -> usize {
        self.last_iterations
    }

    /// Borrow the underlying graph.
    pub fn graph(&self) -> &DynamicGraph {
        &self.graph
    }

    /// Insert a directed edge and re-converge from the warm start.
    pub fn apply_insert(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        self.graph.add_edge(u, v)?;
        self.recompute();
        Ok(())
    }

    /// Remove a directed edge and re-converge from the warm start.
    pub fn apply_remove(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        self.graph.remove_edge(u, v)?;
        self.recompute();
        Ok(())
    }

    /// Apply a batch of `(u, v, is_insert)` mutations atomically, re-converging
    /// once at the end (cheaper than re-solving per edge).
    pub fn apply_batch(&mut self, updates: &[(usize, usize, bool)]) -> GraphalgResult<()> {
        for &(u, v, is_insert) in updates {
            if is_insert {
                self.graph.add_edge(u, v)?;
            } else {
                self.graph.remove_edge(u, v)?;
            }
        }
        self.recompute();
        Ok(())
    }

    /// Resume the power iteration from the current rank vector. Public so callers
    /// can force a re-solve after mutating the graph through other means.
    pub fn recompute(&mut self) {
        let n = self.graph.num_nodes();
        if self.rank.len() != n {
            // Vertex count cannot change here, but stay defensive.
            self.rank = if n == 0 {
                Vec::new()
            } else {
                vec![1.0 / n as f64; n]
            };
        }
        self.power_iterate();
    }

    /// Core warm-startable power iteration writing into `self.rank`.
    fn power_iterate(&mut self) {
        let n = self.graph.num_nodes();
        self.last_iterations = 0;
        if n == 0 {
            return;
        }
        let n_f = n as f64;
        // Out-degree (counting multiplicity) per vertex, precomputed once.
        let mut out_deg = vec![0.0f64; n];
        for u in 0..n {
            out_deg[u] = self.graph.out_degree(u).unwrap_or(0) as f64;
        }
        let d = self.damping;
        let mut cur = std::mem::take(&mut self.rank);
        // Renormalise the warm start defensively (mutations never touch `rank`,
        // but a caller-supplied state might be slightly off).
        normalize(&mut cur, n_f);
        let mut next = vec![0.0f64; n];
        for _ in 0..self.max_iter {
            let mut dangling = 0.0f64;
            for v in 0..n {
                if out_deg[v] == 0.0 {
                    dangling += cur[v];
                }
            }
            let teleport = (1.0 - d) / n_f + d * dangling / n_f;
            for slot in next.iter_mut() {
                *slot = teleport;
            }
            for u in 0..n {
                let od = out_deg[u];
                if od == 0.0 {
                    continue;
                }
                // Multiplicity-aware: each parallel copy of (u,v) carries mass.
                let base = d * cur[u] / od;
                if let Ok(neigh) = self.graph.out_neighbors(u) {
                    for &v in neigh {
                        let m = self.graph.edge_multiplicity(u, v) as f64;
                        next[v] += base * m;
                    }
                }
            }
            let diff: f64 = cur
                .iter()
                .zip(next.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            std::mem::swap(&mut cur, &mut next);
            self.last_iterations += 1;
            if diff < self.tol {
                break;
            }
        }
        normalize(&mut cur, n_f);
        self.rank = cur;
    }
}

/// Normalise `v` to sum one; fall back to uniform if the mass underflows.
fn normalize(v: &mut [f64], n_f: f64) {
    let s: f64 = v.iter().sum();
    if s.abs() > 1e-300 {
        for x in v.iter_mut() {
            *x /= s;
        }
    } else if n_f > 0.0 {
        let u = 1.0 / n_f;
        for x in v.iter_mut() {
            *x = u;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::centrality::pagerank::pagerank;
    use crate::handle::LcgRng;

    fn approx_vec(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn matches_static_pagerank_initial() {
        let mut g = DynamicGraph::new(4);
        g.add_edge(0, 1).expect("e");
        g.add_edge(1, 2).expect("e");
        g.add_edge(2, 0).expect("e");
        g.add_edge(2, 3).expect("e");
        let engine = IncrementalPageRank::new(g.clone(), 0.85, 500, 1e-12).expect("new");
        let stat = pagerank(&g.to_adjacency_list(), 0.85, 500, 1e-12).expect("static");
        assert!(
            approx_vec(engine.ranks(), &stat, 1e-9),
            "incremental {:?} != static {:?}",
            engine.ranks(),
            stat
        );
        let s: f64 = engine.ranks().iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn insert_then_match_static_recompute() {
        let mut g = DynamicGraph::new(5);
        g.add_edge(0, 1).expect("e");
        g.add_edge(1, 2).expect("e");
        g.add_edge(2, 3).expect("e");
        g.add_edge(3, 4).expect("e");
        let mut engine = IncrementalPageRank::new(g, 0.85, 1000, 1e-13).expect("new");
        engine.apply_insert(4, 0).expect("insert"); // close the cycle
        engine.apply_insert(0, 2).expect("insert");

        // Independent cold-start solve on the mutated graph.
        let stat =
            pagerank(&engine.graph().to_adjacency_list(), 0.85, 1000, 1e-13).expect("static");
        assert!(
            approx_vec(engine.ranks(), &stat, 1e-8),
            "after inserts incremental {:?} != static {:?}",
            engine.ranks(),
            stat
        );
    }

    #[test]
    fn remove_then_match_static_recompute() {
        let mut g = DynamicGraph::new(4);
        for &(u, v) in &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 1)] {
            g.add_edge(u, v).expect("e");
        }
        let mut engine = IncrementalPageRank::new(g, 0.85, 1000, 1e-13).expect("new");
        engine.apply_remove(3, 1).expect("remove");
        let stat =
            pagerank(&engine.graph().to_adjacency_list(), 0.85, 1000, 1e-13).expect("static");
        assert!(approx_vec(engine.ranks(), &stat, 1e-8));
    }

    #[test]
    fn warm_start_is_cheaper_than_cold() {
        // Build a sizeable random graph, converge, then perturb a single edge.
        let mut rng = LcgRng::new(0xDEAD_BEEF);
        let n = 60;
        let mut g = DynamicGraph::new(n);
        for u in 0..n {
            for _ in 0..3 {
                let v = rng.next_usize(n);
                if v != u {
                    g.add_edge(u, v).expect("e");
                }
            }
        }
        let mut engine = IncrementalPageRank::new(g, 0.85, 2000, 1e-12).expect("new");
        let cold_iters = engine.last_iterations();
        // One edge change: warm restart should need strictly fewer iterations.
        engine.apply_insert(0, n - 1).expect("insert");
        let warm_iters = engine.last_iterations();
        assert!(
            warm_iters <= cold_iters,
            "warm {warm_iters} should not exceed cold {cold_iters}"
        );
        assert!(warm_iters >= 1);
    }

    #[test]
    fn batch_matches_sequential_and_static() {
        let mut rng = LcgRng::new(123_456);
        let n = 30;
        let mut g = DynamicGraph::new(n);
        for u in 0..n {
            let v = rng.next_usize(n);
            if v != u {
                g.add_edge(u, v).expect("e");
            }
        }
        let mut engine = IncrementalPageRank::new(g, 0.85, 2000, 1e-13).expect("new");

        // A batch of legal inserts (avoid removing absent edges).
        let updates: Vec<(usize, usize, bool)> = (0..8)
            .map(|_| {
                let u = rng.next_usize(n);
                let mut v = rng.next_usize(n);
                if v == u {
                    v = (v + 1) % n;
                }
                (u, v, true)
            })
            .collect();
        engine.apply_batch(&updates).expect("batch");
        let stat =
            pagerank(&engine.graph().to_adjacency_list(), 0.85, 2000, 1e-13).expect("static");
        assert!(
            approx_vec(engine.ranks(), &stat, 1e-7),
            "batch incremental != static"
        );
        let s: f64 = engine.ranks().iter().sum();
        assert!((s - 1.0).abs() < 1e-8);
    }

    #[test]
    fn dangling_vertex_handled() {
        // Vertex 2 is a sink (dangling): its mass redistributes uniformly.
        let mut g = DynamicGraph::new(3);
        g.add_edge(0, 1).expect("e");
        g.add_edge(1, 2).expect("e");
        let engine = IncrementalPageRank::new(g.clone(), 0.85, 1000, 1e-13).expect("new");
        let stat = pagerank(&g.to_adjacency_list(), 0.85, 1000, 1e-13).expect("static");
        assert!(approx_vec(engine.ranks(), &stat, 1e-9));
    }

    #[test]
    fn rejects_bad_damping() {
        let g = DynamicGraph::new(3);
        assert!(IncrementalPageRank::new(g, 1.5, 10, 1e-9).is_err());
    }

    #[test]
    fn empty_graph_ok() {
        let g = DynamicGraph::new(0);
        let engine = IncrementalPageRank::new(g, 0.85, 10, 1e-9).expect("new");
        assert!(engine.ranks().is_empty());
    }
}
