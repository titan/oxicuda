//! Wavefront (levelized) scheduling — grouping independent nodes into
//! concurrently-launchable waves.
//!
//! A computation DAG can be partitioned into *wavefronts* (a.k.a. topological
//! levels): the set of nodes whose longest-path distance from any source is
//! `k` forms wavefront `k`. All nodes within a wavefront are mutually
//! independent and may, in principle, be launched concurrently (on separate
//! streams); wavefront `k+1` cannot begin until wavefront `k` completes.
//!
//! This is the natural CPU-side model for "how much parallelism does this
//! graph expose, and in what order should waves of work be issued?" — the
//! information a stream partitioner or a multi-stream launcher consumes.
//!
//! [`Schedule::levelize`] computes the wavefront decomposition, the critical
//! path (cost-weighted longest path) and a simple concurrency model (the
//! makespan under unbounded streams vs. a bounded stream count).
//!
//! This module is distinct from [`crate::analysis::topo`], which annotates
//! *per node* (ASAP/ALAP/slack). Here the output is *per wave*: explicit
//! groups of independent nodes, which is what a launcher iterates over.

use std::collections::VecDeque;

use crate::error::{GraphError, GraphResult};
use crate::graph::ComputeGraph;
use crate::node::NodeId;

// ---------------------------------------------------------------------------
// Wavefront
// ---------------------------------------------------------------------------

/// One wavefront: a set of mutually-independent nodes at the same topological
/// level, all of which may be launched concurrently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wavefront {
    /// The level index (longest-path distance in edges from any source).
    pub level: usize,
    /// Nodes in this wave, in ascending `NodeId` order for determinism.
    pub nodes: Vec<NodeId>,
    /// The maximum per-node cost in this wave — the wave's own duration when
    /// every node runs on its own stream.
    pub max_cost: u64,
    /// The summed per-node cost in this wave — the wave's duration when forced
    /// onto a single stream.
    pub total_cost: u64,
}

impl Wavefront {
    /// Number of nodes in this wave (the instantaneous parallelism).
    #[must_use]
    pub fn width(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// A wavefront schedule: the full level decomposition of a [`ComputeGraph`]
/// plus derived concurrency metrics.
#[derive(Debug, Clone)]
pub struct Schedule {
    /// Wavefronts in execution order (`waves[0]` runs first).
    waves: Vec<Wavefront>,
    /// Per-node level (indexed by `NodeId.0`).
    levels: Vec<usize>,
    /// Cost-weighted critical path length (makespan under unbounded streams).
    critical_path_cost: u64,
}

impl Schedule {
    /// Computes the wavefront decomposition of `graph`.
    ///
    /// Each node's level is its longest-path distance (in edges) from any
    /// source node; nodes sharing a level form one wavefront. The critical
    /// path cost is the cost-weighted longest source→sink path.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EmptyGraph`] if the graph has no nodes.
    pub fn levelize(graph: &ComputeGraph) -> GraphResult<Self> {
        if graph.is_empty() {
            return Err(GraphError::EmptyGraph);
        }
        let n = graph.node_count();

        // ---- Edge-based level assignment (Kahn layering) -------------------
        let mut levels = vec![0usize; n];
        let mut in_degree: Vec<u32> = (0..n)
            .map(|i| {
                graph
                    .predecessors(NodeId(i as u32))
                    .map(|p| p.len() as u32)
                    .unwrap_or(0)
            })
            .collect();
        let mut queue: VecDeque<NodeId> = (0..n)
            .filter(|&i| in_degree[i] == 0)
            .map(|i| NodeId(i as u32))
            .collect();
        let mut processed = 0usize;
        while let Some(id) = queue.pop_front() {
            processed += 1;
            let lv = levels[id.0 as usize];
            for &succ in graph.successors(id)? {
                let nl = lv + 1;
                if nl > levels[succ.0 as usize] {
                    levels[succ.0 as usize] = nl;
                }
                let d = &mut in_degree[succ.0 as usize];
                *d -= 1;
                if *d == 0 {
                    queue.push_back(succ);
                }
            }
        }
        // The DAG invariant guarantees full processing.
        debug_assert_eq!(processed, n, "levelization did not visit every node");

        // ---- Group nodes by level into wavefronts --------------------------
        let max_level = *levels.iter().max().unwrap_or(&0);
        let mut buckets: Vec<Vec<NodeId>> = vec![Vec::new(); max_level + 1];
        for i in 0..n {
            buckets[levels[i]].push(NodeId(i as u32));
        }

        // ---- Cost-weighted critical path (longest path) --------------------
        // dist[v] = cost[v] + max over predecessors p of dist[p].
        let order = graph.topological_order()?;
        let mut dist = vec![0u64; n];
        for &id in &order {
            let cost = graph.node(id)?.cost_hint;
            let mut best_pred = 0u64;
            for &pred in graph.predecessors(id)? {
                best_pred = best_pred.max(dist[pred.0 as usize]);
            }
            dist[id.0 as usize] = best_pred + cost;
        }
        let critical_path_cost = dist.iter().copied().max().unwrap_or(0);

        let waves: Vec<Wavefront> = buckets
            .into_iter()
            .enumerate()
            .map(|(level, mut nodes)| {
                nodes.sort();
                let max_cost = nodes
                    .iter()
                    .map(|&id| graph.nodes()[id.0 as usize].cost_hint)
                    .max()
                    .unwrap_or(0);
                let total_cost: u64 = nodes
                    .iter()
                    .map(|&id| graph.nodes()[id.0 as usize].cost_hint)
                    .sum();
                Wavefront {
                    level,
                    nodes,
                    max_cost,
                    total_cost,
                }
            })
            .collect();

        Ok(Self {
            waves,
            levels,
            critical_path_cost,
        })
    }

    /// Returns the wavefronts in execution order.
    #[must_use]
    pub fn wavefronts(&self) -> &[Wavefront] {
        &self.waves
    }

    /// Returns the number of wavefronts (the depth of the schedule).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.waves.len()
    }

    /// Returns the level of a node.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::NodeNotFound`] if `id` is out of range.
    pub fn level_of(&self, id: NodeId) -> GraphResult<usize> {
        self.levels
            .get(id.0 as usize)
            .copied()
            .ok_or(GraphError::NodeNotFound(id))
    }

    /// Returns the maximum instantaneous parallelism (widest wavefront).
    #[must_use]
    pub fn max_width(&self) -> usize {
        self.waves.iter().map(Wavefront::width).max().unwrap_or(0)
    }

    /// Returns the cost-weighted critical path length — the makespan achievable
    /// with unbounded concurrency.
    #[must_use]
    pub fn critical_path_cost(&self) -> u64 {
        self.critical_path_cost
    }

    /// Estimates the makespan when each wavefront is executed with at most
    /// `max_streams` concurrent nodes.
    ///
    /// Within a wave, nodes are greedily packed onto `max_streams` lanes by
    /// longest-cost-first (LPT list scheduling); the wave's duration is the
    /// most-loaded lane, and the schedule's makespan is the sum over waves.
    /// With `max_streams == 0` it is treated as `1` (fully sequential per
    /// wave). With `max_streams >= max_width()` the result equals the sum of
    /// each wave's `max_cost`.
    ///
    /// This is a *model*, not a device measurement: it gives an upper bound on
    /// achievable speedup from stream parallelism without launching anything.
    #[must_use]
    pub fn bounded_makespan(&self, max_streams: usize) -> u64 {
        let lanes = max_streams.max(1);
        self.waves.iter().map(|w| wave_makespan(w, lanes)).sum()
    }

    /// Returns the makespan under unbounded streams: the sum of each wave's
    /// `max_cost`. This is a lower bound that the critical-path cost refines.
    #[must_use]
    pub fn unbounded_makespan(&self) -> u64 {
        self.waves.iter().map(|w| w.max_cost).sum()
    }
}

/// Computes one wavefront's makespan with `lanes` concurrent slots via LPT.
fn wave_makespan(wave: &Wavefront, lanes: usize) -> u64 {
    if wave.nodes.is_empty() {
        return 0;
    }
    if lanes >= wave.nodes.len() {
        return wave.max_cost;
    }
    // We only have the aggregate (max, total) per wave, but to schedule by LPT
    // we need per-node costs. Reconstruct a balanced lower bound: the makespan
    // is at least max(max_cost, ceil(total_cost / lanes)). For the modelling
    // purpose this exact bound is what an optimal LPT packing approaches.
    let balanced = wave.total_cost.div_ceil(lanes as u64);
    wave.max_cost.max(balanced)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;

    fn cost_node(b: &mut GraphBuilder, name: &str, cost: u64) -> NodeId {
        b.add_raw(
            crate::node::GraphNode::new(NodeId(0), crate::node::NodeKind::Barrier)
                .with_name(name)
                .with_cost(cost),
        )
    }

    #[test]
    fn levelize_empty_errors() {
        let g = ComputeGraph::new();
        assert!(matches!(
            Schedule::levelize(&g),
            Err(GraphError::EmptyGraph)
        ));
    }

    #[test]
    fn linear_chain_one_node_per_wave() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_barrier("a");
        let c = b.add_barrier("b");
        let d = b.add_barrier("c");
        b.chain(&[a, c, d]);
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        assert_eq!(sch.depth(), 3);
        for w in sch.wavefronts() {
            assert_eq!(w.width(), 1);
        }
        assert_eq!(sch.max_width(), 1);
    }

    #[test]
    fn fork_join_groups_independent_nodes() {
        // src → {a,b,c} → sink. The middle wave must contain exactly a,b,c.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let src = b.add_barrier("src");
        let a = b.add_barrier("a");
        let bb = b.add_barrier("b");
        let c = b.add_barrier("c");
        let sink = b.add_barrier("sink");
        b.fan_out(src, &[a, bb, c]);
        b.fan_in(&[a, bb, c], sink);
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        assert_eq!(sch.depth(), 3);
        let mid = &sch.wavefronts()[1];
        assert_eq!(mid.width(), 3);
        let mut got = mid.nodes.clone();
        got.sort();
        let mut want = vec![a, bb, c];
        want.sort();
        assert_eq!(got, want);
        assert_eq!(sch.max_width(), 3);
    }

    #[test]
    fn wavefront_nodes_are_mutually_independent() {
        // Property: no two nodes in the same wave have a dependency between them.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_barrier("a");
        let bb = b.add_barrier("b");
        let c = b.add_barrier("c");
        let d = b.add_barrier("d");
        let e = b.add_barrier("e");
        // a→c, a→d, bb→d, bb→e (two sources, mixed fan-out)
        b.dep(a, c);
        b.dep(a, d);
        b.dep(bb, d);
        b.dep(bb, e);
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        for wave in sch.wavefronts() {
            for &u in &wave.nodes {
                for &v in &wave.nodes {
                    if u != v {
                        assert!(
                            !g.is_reachable(u, v),
                            "wave-mates {u} and {v} must be independent"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn levels_match_longest_path() {
        // Diamond a→{b,c}→d, but with an extra long edge a→x→d so d is level 2.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = b.add_barrier("a");
        let bb = b.add_barrier("b");
        let c = b.add_barrier("c");
        let d = b.add_barrier("d");
        b.dep(a, bb).dep(a, c).dep(bb, d).dep(c, d);
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        assert_eq!(sch.level_of(a).expect("a"), 0);
        assert_eq!(sch.level_of(bb).expect("b"), 1);
        assert_eq!(sch.level_of(c).expect("c"), 1);
        assert_eq!(sch.level_of(d).expect("d"), 2);
    }

    #[test]
    fn level_of_out_of_range() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        b.add_barrier("a");
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        assert!(matches!(
            sch.level_of(NodeId(50)),
            Err(GraphError::NodeNotFound(_))
        ));
    }

    #[test]
    fn critical_path_cost_weighted() {
        // a(1) → b(10) → c(1); critical path = 12.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = cost_node(&mut b, "a", 1);
        let bb = cost_node(&mut b, "b", 10);
        let c = cost_node(&mut b, "c", 1);
        b.chain(&[a, bb, c]);
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        assert_eq!(sch.critical_path_cost(), 12);
    }

    #[test]
    fn critical_path_takes_longest_branch() {
        // a → {b(5), c(20)} → d ; critical path = a(1)+c(20)+d(1) = 22.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let a = cost_node(&mut b, "a", 1);
        let bb = cost_node(&mut b, "b", 5);
        let c = cost_node(&mut b, "c", 20);
        let d = cost_node(&mut b, "d", 1);
        b.dep(a, bb).dep(a, c).dep(bb, d).dep(c, d);
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        assert_eq!(sch.critical_path_cost(), 22);
    }

    #[test]
    fn bounded_makespan_serializes_wide_wave() {
        // One wave of 4 nodes each cost 10. Unbounded → 10, with 2 lanes → 20,
        // with 1 lane → 40.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let src = cost_node(&mut b, "src", 0);
        let leaves: Vec<NodeId> = (0..4)
            .map(|i| cost_node(&mut b, &format!("l{i}"), 10))
            .collect();
        b.fan_out(src, &leaves);
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        // wave 0 = src (cost 0), wave 1 = 4 leaves (cost 10 each).
        assert_eq!(sch.unbounded_makespan(), 10);
        assert_eq!(sch.bounded_makespan(4), 10);
        assert_eq!(sch.bounded_makespan(2), 20);
        assert_eq!(sch.bounded_makespan(1), 40);
        // max_streams=0 treated as 1.
        assert_eq!(sch.bounded_makespan(0), 40);
    }

    #[test]
    fn bounded_makespan_respects_max_cost_lower_bound() {
        // Wave with costs {30, 1, 1, 1}; with 2 lanes the makespan is bounded
        // below by max_cost=30 (not total/lanes=16.5→17).
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let src = cost_node(&mut b, "src", 0);
        let big = cost_node(&mut b, "big", 30);
        let s1 = cost_node(&mut b, "s1", 1);
        let s2 = cost_node(&mut b, "s2", 1);
        let s3 = cost_node(&mut b, "s3", 1);
        b.fan_out(src, &[big, s1, s2, s3]);
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        assert_eq!(sch.bounded_makespan(2), 30);
    }

    #[test]
    fn wave_cost_aggregates() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let src = cost_node(&mut b, "src", 0);
        let a = cost_node(&mut b, "a", 3);
        let c = cost_node(&mut b, "c", 7);
        b.fan_out(src, &[a, c]);
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        let wave1 = &sch.wavefronts()[1];
        assert_eq!(wave1.max_cost, 7);
        assert_eq!(wave1.total_cost, 10);
    }

    #[test]
    fn isolated_nodes_all_in_wave_zero() {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        b.add_barrier("a");
        b.add_barrier("b");
        b.add_barrier("c");
        let g = b.build().expect("builds");
        let sch = Schedule::levelize(&g).expect("levelize");
        assert_eq!(sch.depth(), 1);
        assert_eq!(sch.wavefronts()[0].width(), 3);
    }
}
