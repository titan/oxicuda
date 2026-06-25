//! Reduction-pattern fusion pass — fuses reduction-chained element-wise
//! regions (`LayerNorm`, `softmax`, …) that the linear element-wise fuser in
//! [`crate::optimizer::fusion`] cannot capture.
//!
//! # Why a separate pass
//!
//! The element-wise chain fuser merges only *linear* producer → consumer
//! chains in which every fused node is the **sole** consumer of its
//! predecessor. Reduction patterns break that rule on purpose:
//!
//! ```text
//!   LayerNorm:                       Softmax:
//!
//!        x                               x
//!        │                               │
//!     ┌──┴── mean (reduction)        ┌───┴── max (reduction)
//!     │       │                      │        │
//!     ▼       ▼                      ▼        ▼
//!     sub ◄───┘  (broadcast)         sub ◄────┘  (broadcast)
//!     │                               │
//!     ▼                               ▼
//!     var (reduction)                exp
//!     │                               │
//!     ▼                               ▼
//!     normalize ◄─ (broadcast)       sum (reduction)
//!     │                               │
//!     ▼                               ▼
//!     scale_shift                    divide ◄─ (broadcast)
//! ```
//!
//! A reduction node produces a **small** statistic that is *broadcast* back to
//! one or more element-wise consumers, so the reduction output fans out to
//! several nodes (out-degree ≥ 2) and the region re-converges at a single
//! sink. That diamond shape is precisely what the chain fuser rejects (it
//! treats any fan-out as "parallel branches, do not fuse"). Fusing these
//! regions into one kernel removes the intermediate broadcast round-trips to
//! device memory — the dominant cost of an un-fused `LayerNorm`/`softmax`.
//!
//! # What is fused
//!
//! A set of nodes is a **reduction-fusion region** rooted at a fusible kernel
//! `r` when:
//!
//! 1. `r` is a fusible [`NodeKind::KernelLaunch`].
//! 2. The region `R = {r} ∪ members` forms a **single-entry, single-exit**
//!    (SESE) subgraph: `r` dominates every member, and a unique member `sink`
//!    is reachable from every member and is the only node in `R` with a
//!    successor outside `R` (post-dominates the region).
//! 3. Every member is a fusible kernel with a launch configuration compatible
//!    with `r` (same total thread count).
//! 4. The region is **closed**: no edge leaves a non-`sink` member to a node
//!    outside `R`. This is what makes the rewrite semantics-preserving — every
//!    intermediate buffer is consumed entirely inside the fused kernel and is
//!    never observed by the rest of the graph.
//! 5. The region contains genuine **broadcast fan-out**: at least one member
//!    has out-degree ≥ 2 *inside* `R` (the reduction → broadcast shape). This
//!    is what distinguishes a reduction region from a plain linear chain
//!    (handled by the element-wise fuser) and forces region size ≥ 3.
//!
//! When a region matches, the pass emits a [`ReductionFusionGroup`] describing
//! the merge. As with [`crate::optimizer::fusion`], the graph itself is **not**
//! mutated — the descriptor is consumed downstream by the PTX codegen layer,
//! and the [`rewrite`] helper materialises the fused [`ComputeGraph`] when a
//! concrete rewritten DAG is required (e.g. for the CPU simulator).
//!
//! # Algorithm
//!
//! 1. Topological + dominance analysis.
//! 2. For each fusible kernel `r` in topological order that is not yet claimed,
//!    grow the dominated, fusible, closed region below `r` and test the SESE +
//!    fan-out conditions.
//! 3. Accept the largest such region; mark its members claimed so they are not
//!    re-used by a later (smaller) region.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::analysis::{dominance_analyse, topo_analyse};
use crate::error::{GraphError, GraphResult};
use crate::graph::ComputeGraph;
use crate::node::{BufferId, GraphNode, KernelConfig, NodeId, NodeKind};

// ---------------------------------------------------------------------------
// ReductionPattern
// ---------------------------------------------------------------------------

/// The classified shape of a fused reduction region.
///
/// Classification is a *descriptive* label derived from the member kernel
/// function names; it never affects whether a region is fusible (that decision
/// is purely structural). It is used to tag the fused kernel and to let the
/// codegen layer pick a specialised template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionPattern {
    /// `mean → subtract → variance → normalize → scale/shift` (two reductions).
    LayerNorm,
    /// `max → subtract → exp → sum → divide` (two reductions, one exp).
    Softmax,
    /// A reduction-broadcast region that does not match a known template.
    Generic,
}

impl ReductionPattern {
    /// Returns a short, stable identifier for the pattern.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::LayerNorm => "layernorm",
            Self::Softmax => "softmax",
            Self::Generic => "reduction",
        }
    }
}

impl std::fmt::Display for ReductionPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// ReductionFusionGroup
// ---------------------------------------------------------------------------

/// A reduction-broadcast region that the pass merges into a single kernel.
///
/// Members are listed in topological order (the rooting reduction first, the
/// post-dominating sink last).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReductionFusionGroup {
    /// Group identifier (sequential, 0-based).
    pub id: usize,
    /// The rooting reduction node (single entry of the region).
    pub root: NodeId,
    /// The post-dominating exit node (single exit of the region).
    pub sink: NodeId,
    /// All region members in topological order (`root` first, `sink` last).
    pub members: Vec<NodeId>,
    /// Classified pattern shape.
    pub pattern: ReductionPattern,
    /// Combined launch configuration (uses the root's config).
    pub config: KernelConfig,
    /// Human-readable tag for debugging.
    pub tag: String,
}

impl ReductionFusionGroup {
    /// Returns the number of nodes merged by this region.
    #[must_use]
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Returns the number of kernel launches this region eliminates
    /// (`size - 1`, since the whole region collapses to one launch).
    #[must_use]
    pub fn launches_saved(&self) -> usize {
        self.members.len().saturating_sub(1)
    }
}

// ---------------------------------------------------------------------------
// ReductionFusionPlan
// ---------------------------------------------------------------------------

/// The complete reduction-fusion plan produced by [`analyse`].
#[derive(Debug, Clone, Default)]
pub struct ReductionFusionPlan {
    /// All matched reduction-fusion regions.
    pub groups: Vec<ReductionFusionGroup>,
    /// Map from a region member [`NodeId`] to its group index.
    pub node_to_group: HashMap<NodeId, usize>,
}

impl ReductionFusionPlan {
    /// Returns the number of matched reduction regions.
    #[must_use]
    pub fn fusion_count(&self) -> usize {
        self.groups.len()
    }

    /// Returns the total number of kernel launches eliminated across all
    /// regions.
    #[must_use]
    pub fn nodes_saved(&self) -> usize {
        self.groups
            .iter()
            .map(ReductionFusionGroup::launches_saved)
            .sum()
    }

    /// Returns the region that owns `node`, if any.
    #[must_use]
    pub fn group_of(&self, node: NodeId) -> Option<&ReductionFusionGroup> {
        self.node_to_group
            .get(&node)
            .and_then(|&idx| self.groups.get(idx))
    }

    /// Returns `true` if `node` was absorbed into a fused region but is not
    /// that region's root (i.e. it disappears as an independent launch).
    #[must_use]
    pub fn is_absorbed(&self, node: NodeId) -> bool {
        match self.group_of(node) {
            Some(g) => g.root != node,
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `(fusible, config)` if `node` is a kernel launch, else `None`.
fn kernel_meta(graph: &ComputeGraph, node: NodeId) -> Option<(bool, KernelConfig)> {
    match &graph.node(node).ok()?.kind {
        NodeKind::KernelLaunch {
            fusible, config, ..
        } => Some((*fusible, *config)),
        _ => None,
    }
}

/// Returns the lowercased function name of a kernel node (empty if not a kernel).
fn fn_name_lower(graph: &ComputeGraph, node: NodeId) -> String {
    graph
        .node(node)
        .ok()
        .and_then(|n| n.kind.function_name())
        .unwrap_or("")
        .to_ascii_lowercase()
}

/// Two configs fuse iff they launch the same total number of threads.
fn configs_compatible(a: &KernelConfig, b: &KernelConfig) -> bool {
    a.total_threads() == b.total_threads()
}

/// Classifies a region's pattern from the member function names.
///
/// Recognises the canonical `LayerNorm` and `softmax` op vocabularies; any
/// other reduction-broadcast region is [`ReductionPattern::Generic`].
fn classify(graph: &ComputeGraph, members: &[NodeId]) -> ReductionPattern {
    let names: Vec<String> = members.iter().map(|&m| fn_name_lower(graph, m)).collect();
    let has = |needle: &str| names.iter().any(|n| n.contains(needle));

    // Softmax fingerprint: an exponential plus a normalising divide/sum.
    let softmax_like = has("exp") && (has("softmax") || has("div") || has("sum") || has("norm"));
    // LayerNorm fingerprint: mean/variance statistics feeding a normalise.
    let layernorm_like = (has("mean") || has("avg"))
        && (has("var") || has("std") || has("rms") || has("norm") || has("layernorm"));

    if has("softmax") || softmax_like {
        ReductionPattern::Softmax
    } else if has("layernorm") || layernorm_like {
        ReductionPattern::LayerNorm
    } else {
        ReductionPattern::Generic
    }
}

// ---------------------------------------------------------------------------
// Region growth
// ---------------------------------------------------------------------------

/// Attempts to grow a maximal single-entry/single-exit, closed, fusible
/// reduction region rooted at `root`.
///
/// Returns the region's members (topologically sorted, `root` first, `sink`
/// last) on success, or `None` if no valid region exists.
fn grow_region(
    graph: &ComputeGraph,
    root: NodeId,
    dt: &crate::analysis::DomTree,
    topo_pos: &HashMap<NodeId, usize>,
    claimed: &HashSet<NodeId>,
) -> GraphResult<Option<Vec<NodeId>>> {
    let root_config = match kernel_meta(graph, root) {
        Some((true, cfg)) => cfg,
        _ => return Ok(None),
    };

    // Candidate members: nodes dominated by `root` that are fusible kernels
    // with a compatible config, reachable from `root`, and not already claimed.
    // We grow the region by forward BFS from `root`, only descending through
    // nodes that satisfy the membership predicate, and bound the frontier so a
    // single rogue non-fusible successor terminates that branch.
    let member_ok = |n: NodeId| -> bool {
        if n == root {
            return true;
        }
        if claimed.contains(&n) {
            return false;
        }
        if !dt.dominates(root, n) {
            return false;
        }
        match kernel_meta(graph, n) {
            Some((fusible, cfg)) => fusible && configs_compatible(&root_config, &cfg),
            None => false,
        }
    };

    // Collect the dominated fusible cone reachable from `root` through members.
    let mut region: HashSet<NodeId> = HashSet::new();
    region.insert(root);
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    queue.push_back(root);
    while let Some(cur) = queue.pop_front() {
        for &succ in graph.successors(cur)? {
            if region.contains(&succ) {
                continue;
            }
            if member_ok(succ) {
                region.insert(succ);
                queue.push_back(succ);
            }
        }
    }

    if region.len() < 3 {
        // Too small to be a reduction pattern (a 2-node chain is the
        // element-wise fuser's job).
        return Ok(None);
    }

    // The region's exits are members with at least one successor outside the
    // region. For a SESE region there must be exactly one such exit, and it
    // must be reachable from every member (post-dominator of the region).
    let mut exits: Vec<NodeId> = Vec::new();
    for &m in &region {
        let leaves = graph.successors(m)?.iter().any(|s| !region.contains(s));
        let is_graph_sink = graph.successors(m)?.is_empty();
        if leaves || is_graph_sink {
            exits.push(m);
        }
    }

    // A closed SESE region has a unique exit. If several members leak out of
    // the region, shrink the region to the sub-cone that re-converges: drop
    // every member that is not on a path to a *single* common sink.
    //
    // Strategy: pick the topologically-last member as the candidate sink; then
    // verify (a) it is reachable from every member, and (b) every member's
    // out-of-region successors are absent (closedness). If verification fails,
    // the region is not a clean reduction pattern and we reject it. This keeps
    // the rewrite provably semantics-preserving rather than guessing.
    let sink = *region
        .iter()
        .max_by_key(|&&m| topo_pos.get(&m).copied().unwrap_or(0))
        .ok_or_else(|| GraphError::Internal("reduction region unexpectedly empty".into()))?;

    // (a) Every member must reach the sink (sink post-dominates the region
    //     *within the region's own edges*).
    if !region_reaches_all(graph, &region, sink)? {
        return Ok(None);
    }

    // (b) Closedness: the only member allowed to have successors outside the
    //     region is the sink. Any other leak would expose an intermediate
    //     buffer and break semantics preservation if fused.
    for &m in &region {
        if m == sink {
            continue;
        }
        let leaks = graph.successors(m)?.iter().any(|s| !region.contains(s));
        if leaks {
            return Ok(None);
        }
    }

    // (c) Genuine broadcast fan-out: some member must fan out to ≥ 2 members
    //     inside the region. Without this the region is a plain linear chain.
    let has_fanout = region
        .iter()
        .try_fold(false, |acc, &m| -> GraphResult<bool> {
            if acc {
                return Ok(true);
            }
            let in_region_succ = graph
                .successors(m)?
                .iter()
                .filter(|s| region.contains(s))
                .count();
            Ok(in_region_succ >= 2)
        })?;
    if !has_fanout {
        return Ok(None);
    }

    // Emit members in topological order.
    let mut members: Vec<NodeId> = region.into_iter().collect();
    members.sort_by_key(|m| topo_pos.get(m).copied().unwrap_or(usize::MAX));
    Ok(Some(members))
}

/// Returns `true` if every node in `region` can reach `sink` using only edges
/// that stay inside `region`.
fn region_reaches_all(
    graph: &ComputeGraph,
    region: &HashSet<NodeId>,
    sink: NodeId,
) -> GraphResult<bool> {
    // Reverse BFS from `sink` over in-region edges; everything in `region`
    // must be visited.
    let mut reached: HashSet<NodeId> = HashSet::new();
    reached.insert(sink);
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    queue.push_back(sink);
    while let Some(cur) = queue.pop_front() {
        for &pred in graph.predecessors(cur)? {
            if region.contains(&pred) && reached.insert(pred) {
                queue.push_back(pred);
            }
        }
    }
    Ok(region.iter().all(|m| reached.contains(m)))
}

// ---------------------------------------------------------------------------
// analyse — entry point
// ---------------------------------------------------------------------------

/// Runs the reduction-fusion analysis pass on `graph`.
///
/// Returns a [`ReductionFusionPlan`] describing every reduction-broadcast
/// region that can be merged into a single fused kernel. The input graph is
/// left unchanged.
///
/// # Errors
///
/// Returns [`GraphError::EmptyGraph`] if the graph has no nodes.
pub fn analyse(graph: &ComputeGraph) -> GraphResult<ReductionFusionPlan> {
    if graph.is_empty() {
        return Err(GraphError::EmptyGraph);
    }

    let topo = topo_analyse(graph)?;
    let dt = dominance_analyse(graph)?;
    let topo_pos: HashMap<NodeId, usize> = topo
        .order
        .iter()
        .enumerate()
        .map(|(p, &id)| (id, p))
        .collect();

    let mut claimed: HashSet<NodeId> = HashSet::new();
    let mut groups: Vec<ReductionFusionGroup> = Vec::new();
    let mut node_to_group: HashMap<NodeId, usize> = HashMap::new();

    for &root in &topo.order {
        if claimed.contains(&root) {
            continue;
        }
        // Only fusible kernels can root a region.
        match kernel_meta(graph, root) {
            Some((true, _)) => {}
            _ => continue,
        }

        let members = match grow_region(graph, root, &dt, &topo_pos, &claimed)? {
            Some(m) => m,
            None => continue,
        };

        let sink = *members.last().ok_or_else(|| {
            GraphError::Internal("reduction region members unexpectedly empty".into())
        })?;
        let pattern = classify(graph, &members);
        let config = kernel_meta(graph, root)
            .map(|(_, c)| c)
            .unwrap_or_else(|| KernelConfig::linear(1, 1, 0));

        let gid = groups.len();
        let tag = format!(
            "fused_{}_{}..{}",
            pattern.name(),
            graph.node(root)?.display_name(),
            graph.node(sink)?.display_name()
        );

        for &m in &members {
            claimed.insert(m);
            node_to_group.insert(m, gid);
        }

        groups.push(ReductionFusionGroup {
            id: gid,
            root,
            sink,
            members,
            pattern,
            config,
            tag,
        });
    }

    Ok(ReductionFusionPlan {
        groups,
        node_to_group,
    })
}

// ---------------------------------------------------------------------------
// rewrite — materialise the fused graph
// ---------------------------------------------------------------------------

/// Rewrites `graph` by collapsing every region in `plan` into a single fused
/// kernel node, producing a new [`ComputeGraph`].
///
/// The rewrite is **semantics-preserving**:
///
/// * Each region `R` is replaced by one fused [`NodeKind::KernelLaunch`] whose
///   `inputs` are the buffers read from *outside* `R` and whose `outputs` are
///   the buffers the sink writes (the only buffers observed downstream — the
///   region's closedness guarantees no other member output escapes).
/// * Every edge `p → m` from outside into a member is rerouted to the fused
///   node; every edge `sink → s` out of the region is rerouted from the fused
///   node. Intra-region edges vanish (they are now internal to one kernel).
///
/// Non-region nodes are copied verbatim. Buffer descriptors are preserved.
///
/// # Errors
///
/// Propagates [`GraphError`] from node/edge reconstruction (e.g. a cycle, which
/// cannot occur for a valid SESE collapse but is checked defensively).
pub fn rewrite(graph: &ComputeGraph, plan: &ReductionFusionPlan) -> GraphResult<ComputeGraph> {
    let mut out = ComputeGraph::new();

    // Copy buffer descriptors verbatim (ids are dense and stable).
    for buf in graph.buffers() {
        out.add_buffer(buf.clone());
    }

    // Map every old node to the new node that represents it. Region members
    // all map to their region's single fused node.
    let mut old_to_new: HashMap<NodeId, NodeId> = HashMap::new();

    // 1. Emit non-region nodes and region roots (as fused nodes), in the
    //    original insertion order so ids stay deterministic.
    for old in graph.nodes() {
        let oid = old.id;
        if let Some(group) = plan.group_of(oid) {
            if group.root != oid {
                // Absorbed member: skip; it maps to the fused node later.
                continue;
            }
            // Build the fused node for this region.
            let region: HashSet<NodeId> = group.members.iter().copied().collect();

            // Inputs: buffers read by any member that are produced outside the
            // region (or are graph inputs). Outputs: buffers written by the
            // sink (the externally-visible result).
            let mut region_outputs: HashSet<BufferId> = HashSet::new();
            for &m in &group.members {
                for &b in &graph.node(m)?.outputs {
                    region_outputs.insert(b);
                }
            }
            let mut fused_inputs: Vec<BufferId> = Vec::new();
            let mut seen_in: HashSet<BufferId> = HashSet::new();
            for &m in &group.members {
                for &b in &graph.node(m)?.inputs {
                    // An input is external if no member writes it.
                    if !region_outputs.contains(&b) && seen_in.insert(b) {
                        fused_inputs.push(b);
                    }
                }
            }
            // Outputs visible downstream = the sink's outputs.
            let fused_outputs: Vec<BufferId> = graph.node(group.sink)?.outputs.clone();

            let fn_name = format!(
                "{}_{}",
                group.pattern.name(),
                group
                    .members
                    .iter()
                    .filter_map(|&m| graph.node(m).ok().and_then(|n| n.kind.function_name()))
                    .collect::<Vec<_>>()
                    .join("_")
            );
            let cost: u64 = group
                .members
                .iter()
                .filter_map(|&m| graph.node(m).ok().map(|n| n.cost_hint))
                .sum();
            let kind = NodeKind::KernelLaunch {
                function_name: fn_name,
                config: group.config,
                fusible: true,
            };
            let node = GraphNode::new(NodeId(0), kind)
                .with_inputs(fused_inputs)
                .with_outputs(fused_outputs)
                .with_cost(cost.max(1))
                .with_name(group.tag.clone());
            let nid = out.add_node(node);
            for &m in &region {
                old_to_new.insert(m, nid);
            }
        } else {
            // Plain node: clone its kind/buffers; id is reassigned by add_node.
            let mut node = GraphNode::new(NodeId(0), old.kind.clone())
                .with_inputs(old.inputs.iter().copied())
                .with_outputs(old.outputs.iter().copied())
                .with_cost(old.cost_hint);
            if let Some(s) = old.stream_hint {
                node = node.with_stream(s);
            }
            if let Some(name) = &old.name {
                node = node.with_name(name.clone());
            }
            let nid = out.add_node(node);
            old_to_new.insert(oid, nid);
        }
    }

    // 2. Recreate edges, skipping intra-region edges and de-duplicating.
    let mut added: HashSet<(NodeId, NodeId)> = HashSet::new();
    for (from_old, to_old) in graph.edges() {
        let from_new = *old_to_new
            .get(&from_old)
            .ok_or_else(|| GraphError::Internal("missing node mapping (from)".into()))?;
        let to_new = *old_to_new
            .get(&to_old)
            .ok_or_else(|| GraphError::Internal("missing node mapping (to)".into()))?;
        if from_new == to_new {
            // Intra-region edge collapsed into the fused node.
            continue;
        }
        if added.insert((from_new, to_new)) {
            out.add_edge(from_new, to_new)?;
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::executor::{ExecutionPlan, SequentialExecutor};
    use crate::node::MemcpyDir;

    // -- Builders for canonical reduction patterns -------------------------

    /// Builds a LayerNorm-style region:
    /// `x → mean`, then `{x, mean} → sub`, `sub → var`, `{sub, var} → norm`,
    /// `norm → scale_shift`. The mean and variance reductions each broadcast
    /// to an element-wise consumer; the region re-converges at `scale_shift`.
    ///
    /// Returns `(graph, [mean, sub, var, norm, scale_shift])`.
    fn build_layernorm() -> (ComputeGraph, Vec<NodeId>) {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let mean = b.add_kernel("mean", 4, 256, 0).fusible(true).finish();
        let sub = b.add_kernel("subtract", 4, 256, 0).fusible(true).finish();
        let var = b.add_kernel("variance", 4, 256, 0).fusible(true).finish();
        let norm = b.add_kernel("normalize", 4, 256, 0).fusible(true).finish();
        let scale = b
            .add_kernel("scale_shift", 4, 256, 0)
            .fusible(true)
            .finish();

        // mean broadcasts to sub; sub feeds var; var broadcasts to norm.
        b.dep(mean, sub);
        b.dep(sub, var);
        b.dep(sub, norm); // mean's centred value also flows forward (fan-out of sub)
        b.dep(var, norm);
        b.dep(norm, scale);
        let g = b.build().expect("layernorm graph builds");
        (g, vec![mean, sub, var, norm, scale])
    }

    /// Builds a softmax-style region:
    /// `max → sub`, `sub → exp`, `exp → sum`, `{exp, sum} → div`. The `max`
    /// reduction fans out (to `sub` and onward) and `sum` broadcasts to `div`.
    /// Region re-converges at `div`.
    ///
    /// Returns `(graph, [mx, sub, exp, sum, div])`.
    fn build_softmax() -> (ComputeGraph, Vec<NodeId>) {
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let mx = b.add_kernel("max", 4, 256, 0).fusible(true).finish();
        let sub = b.add_kernel("subtract", 4, 256, 0).fusible(true).finish();
        let exp = b.add_kernel("exp", 4, 256, 0).fusible(true).finish();
        let sum = b.add_kernel("sum", 4, 256, 0).fusible(true).finish();
        let div = b.add_kernel("divide", 4, 256, 0).fusible(true).finish();

        b.dep(mx, sub);
        b.dep(sub, exp);
        b.dep(exp, sum);
        b.dep(exp, div); // exp fans out: feeds both sum and the final divide
        b.dep(sum, div);
        let g = b.build().expect("softmax graph builds");
        (g, vec![mx, sub, exp, sum, div])
    }

    // -- Detection ---------------------------------------------------------

    #[test]
    fn reduction_empty_graph_errors() {
        let g = ComputeGraph::new();
        assert!(matches!(analyse(&g), Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn layernorm_region_detected() {
        let (g, ids) = build_layernorm();
        let plan = analyse(&g).expect("reduction fusion analysis succeeds");
        assert_eq!(plan.fusion_count(), 1);
        let group = plan
            .group_of(ids[0])
            .expect("mean belongs to a fused region");
        // All five nodes are merged.
        assert_eq!(group.size(), 5);
        for id in &ids {
            assert!(group.members.contains(id), "member {id} missing");
        }
        assert_eq!(group.root, ids[0]); // mean is the entry
        assert_eq!(group.sink, ids[4]); // scale_shift is the exit
        assert_eq!(group.pattern, ReductionPattern::LayerNorm);
        // 5 nodes → one launch, saving 4.
        assert_eq!(plan.nodes_saved(), 4);
    }

    #[test]
    fn softmax_region_detected() {
        let (g, ids) = build_softmax();
        let plan = analyse(&g).expect("reduction fusion analysis succeeds");
        assert_eq!(plan.fusion_count(), 1);
        let group = plan
            .group_of(ids[2])
            .expect("exp belongs to a fused region");
        assert_eq!(group.size(), 5);
        assert_eq!(group.root, ids[0]); // max
        assert_eq!(group.sink, ids[4]); // divide
        assert_eq!(group.pattern, ReductionPattern::Softmax);
        assert_eq!(plan.nodes_saved(), 4);
    }

    #[test]
    fn absorbed_members_flagged() {
        let (g, ids) = build_softmax();
        let plan = analyse(&g).expect("reduction fusion analysis succeeds");
        // The root is not "absorbed"; every other member is.
        assert!(!plan.is_absorbed(ids[0]));
        for id in &ids[1..] {
            assert!(plan.is_absorbed(*id), "member {id} should be absorbed");
        }
    }

    // -- Negative cases ----------------------------------------------------

    #[test]
    fn linear_chain_not_a_reduction_region() {
        // A pure linear chain has no broadcast fan-out → left to the
        // element-wise fuser, not matched here.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let k0 = b.add_kernel("a", 4, 256, 0).fusible(true).finish();
        let k1 = b.add_kernel("b", 4, 256, 0).fusible(true).finish();
        let k2 = b.add_kernel("c", 4, 256, 0).fusible(true).finish();
        b.chain(&[k0, k1, k2]);
        let g = b.build().expect("chain graph builds");
        let plan = analyse(&g).expect("reduction fusion analysis succeeds");
        assert_eq!(plan.fusion_count(), 0);
        assert_eq!(plan.nodes_saved(), 0);
    }

    #[test]
    fn non_fusible_member_breaks_region() {
        // Same softmax shape, but `exp` is non-fusible: the closed region can
        // no longer span the reduction, so nothing is fused.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let mx = b.add_kernel("max", 4, 256, 0).fusible(true).finish();
        let sub = b.add_kernel("subtract", 4, 256, 0).fusible(true).finish();
        let exp = b.add_kernel("exp", 4, 256, 0).fusible(false).finish();
        let sum = b.add_kernel("sum", 4, 256, 0).fusible(true).finish();
        let div = b.add_kernel("divide", 4, 256, 0).fusible(true).finish();
        b.dep(mx, sub);
        b.dep(sub, exp);
        b.dep(exp, sum);
        b.dep(exp, div);
        b.dep(sum, div);
        let g = b.build().expect("graph builds");
        let plan = analyse(&g).expect("reduction fusion analysis succeeds");
        assert_eq!(plan.fusion_count(), 0);
    }

    #[test]
    fn open_region_leaking_intermediate_not_fused() {
        // The fan-out node `exp` ALSO feeds an external sink, so every closed
        // reduction diamond would have to expose `exp`'s intermediate buffer.
        // No closed SESE region survives → nothing is fused. (Leaking a
        // *non-fan-out* node such as `sub` would still permit the independent
        // `exp → {sum, div} → div` tail diamond to fuse; closedness is checked
        // per candidate region, so the pass only refuses the regions that would
        // actually hide an observed buffer.)
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let mx = b.add_kernel("max", 4, 256, 0).fusible(true).finish();
        let sub = b.add_kernel("subtract", 4, 256, 0).fusible(true).finish();
        let exp = b.add_kernel("exp", 4, 256, 0).fusible(true).finish();
        let sum = b.add_kernel("sum", 4, 256, 0).fusible(true).finish();
        let div = b.add_kernel("divide", 4, 256, 0).fusible(true).finish();
        // External consumer of the fan-out node's intermediate result.
        let leak = b.add_memcpy("leak", MemcpyDir::DeviceToHost, 1024);
        b.dep(mx, sub);
        b.dep(sub, exp);
        b.dep(exp, sum);
        b.dep(exp, div);
        b.dep(sum, div);
        b.dep(exp, leak); // exp (the fan-out) leaks out of every candidate region
        let g = b.build().expect("graph builds");
        let plan = analyse(&g).expect("reduction fusion analysis succeeds");
        assert_eq!(plan.fusion_count(), 0);
    }

    #[test]
    fn incompatible_config_member_excluded() {
        // `div` has a different total-thread count → cannot join the region;
        // the remaining nodes lose their single exit, so nothing fuses.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let mx = b.add_kernel("max", 4, 256, 0).fusible(true).finish();
        let sub = b.add_kernel("subtract", 4, 256, 0).fusible(true).finish();
        let exp = b.add_kernel("exp", 4, 256, 0).fusible(true).finish();
        let sum = b.add_kernel("sum", 4, 256, 0).fusible(true).finish();
        let div = b.add_kernel("divide", 8, 256, 0).fusible(true).finish(); // 2048 threads
        b.dep(mx, sub);
        b.dep(sub, exp);
        b.dep(exp, sum);
        b.dep(exp, div);
        b.dep(sum, div);
        let g = b.build().expect("graph builds");
        let plan = analyse(&g).expect("reduction fusion analysis succeeds");
        // div is excluded; without it exp fans out to sum only (no re-converge
        // sink inside the region) so the region is rejected.
        assert_eq!(plan.fusion_count(), 0);
    }

    // -- Rewrite + semantics preservation ----------------------------------

    #[test]
    fn rewrite_collapses_region_to_one_node() {
        let (g, _ids) = build_layernorm();
        let plan = analyse(&g).expect("analysis succeeds");
        let fused = rewrite(&g, &plan).expect("rewrite succeeds");
        // 5 nodes collapse to 1.
        assert_eq!(g.node_count(), 5);
        assert_eq!(fused.node_count(), 1);
        // The single node is a fused kernel.
        let only = fused.node(NodeId(0)).expect("fused node exists");
        assert!(only.kind.is_compute());
        assert!(
            only.kind
                .function_name()
                .unwrap_or("")
                .contains("layernorm")
        );
    }

    #[test]
    fn rewrite_preserves_external_topology() {
        // upload → [softmax region] → download. After rewrite the fused node
        // sits between upload and download with the edges preserved.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let up = b.add_memcpy("up", MemcpyDir::HostToDevice, 1024);
        let mx = b.add_kernel("max", 4, 256, 0).fusible(true).finish();
        let sub = b.add_kernel("subtract", 4, 256, 0).fusible(true).finish();
        let exp = b.add_kernel("exp", 4, 256, 0).fusible(true).finish();
        let sum = b.add_kernel("sum", 4, 256, 0).fusible(true).finish();
        let div = b.add_kernel("divide", 4, 256, 0).fusible(true).finish();
        let dn = b.add_memcpy("dn", MemcpyDir::DeviceToHost, 1024);
        b.dep(up, mx);
        b.dep(mx, sub);
        b.dep(sub, exp);
        b.dep(exp, sum);
        b.dep(exp, div);
        b.dep(sum, div);
        b.dep(div, dn);
        let g = b.build().expect("graph builds");
        let plan = analyse(&g).expect("analysis succeeds");
        assert_eq!(plan.fusion_count(), 1);
        let fused = rewrite(&g, &plan).expect("rewrite succeeds");
        // up + fused + dn = 3 nodes.
        assert_eq!(fused.node_count(), 3);
        // upload still reaches download.
        let up_new = fused.sources();
        assert_eq!(up_new.len(), 1);
        let dn_new = fused.sinks();
        assert_eq!(dn_new.len(), 1);
        assert!(fused.is_reachable(up_new[0], dn_new[0]));
        // Exactly one compute node remains.
        assert_eq!(fused.kernel_nodes().len(), 1);
    }

    #[test]
    fn rewrite_no_match_is_identity() {
        // A graph with no reduction region is rewritten unchanged.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let k0 = b.add_kernel("a", 4, 256, 0).fusible(true).finish();
        let k1 = b.add_kernel("b", 4, 256, 0).fusible(true).finish();
        b.chain(&[k0, k1]);
        let g = b.build().expect("graph builds");
        let plan = analyse(&g).expect("analysis succeeds");
        let fused = rewrite(&g, &plan).expect("rewrite succeeds");
        assert_eq!(fused.node_count(), g.node_count());
        assert_eq!(fused.edge_count(), g.edge_count());
    }

    #[test]
    fn simulator_agrees_before_and_after_fusion() {
        // Build a full pipeline: upload → softmax region → download, then check
        // the CPU simulator produces a *semantically equivalent* execution:
        // identical bytes moved, identical reductions of kernel work, and the
        // fused graph never launches more kernels than the original.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let up = b.add_memcpy("up", MemcpyDir::HostToDevice, 4096);
        let mx = b.add_kernel("max", 4, 256, 0).fusible(true).finish();
        let sub = b.add_kernel("subtract", 4, 256, 0).fusible(true).finish();
        let exp = b.add_kernel("exp", 4, 256, 0).fusible(true).finish();
        let sum = b.add_kernel("sum", 4, 256, 0).fusible(true).finish();
        let div = b.add_kernel("divide", 4, 256, 0).fusible(true).finish();
        let dn = b.add_memcpy("dn", MemcpyDir::DeviceToHost, 4096);
        b.dep(up, mx);
        b.dep(mx, sub);
        b.dep(sub, exp);
        b.dep(exp, sum);
        b.dep(exp, div);
        b.dep(sum, div);
        b.dep(div, dn);
        let g = b.build().expect("graph builds");

        let plan = analyse(&g).expect("analysis succeeds");
        assert_eq!(plan.fusion_count(), 1);
        let fused = rewrite(&g, &plan).expect("rewrite succeeds");

        let before =
            SequentialExecutor::new(&ExecutionPlan::build(&g, 4).expect("plan(before) builds"))
                .run()
                .expect("before runs");
        let after =
            SequentialExecutor::new(&ExecutionPlan::build(&fused, 4).expect("plan(after) builds"))
                .run()
                .expect("after runs");

        // Memory traffic is identical (the data semantics are unchanged):
        // the same bytes are uploaded and downloaded before and after fusion.
        assert_eq!(before.bytes_copied, after.bytes_copied);
        assert_eq!(before.bytes_copied, 4096 * 2);
        assert_eq!(before.bytes_set, after.bytes_set);
        // The whole reduction region collapses to a single launch.
        assert_eq!(after.kernels_launched, 1);
        // Reduction fusion never *increases* the launch count over the
        // baseline plan (which has already run the element-wise chain fuser).
        assert!(after.kernels_launched <= before.kernels_launched);
    }

    #[test]
    fn pattern_display_and_name() {
        assert_eq!(ReductionPattern::LayerNorm.name(), "layernorm");
        assert_eq!(ReductionPattern::Softmax.to_string(), "softmax");
        assert_eq!(ReductionPattern::Generic.name(), "reduction");
    }

    #[test]
    fn generic_reduction_region_classified() {
        // A reduction-broadcast diamond whose names match no known template.
        let mut b = GraphBuilder::new().with_auto_infer_edges(false);
        let r = b.add_kernel("reduce_op", 4, 256, 0).fusible(true).finish();
        let a = b.add_kernel("elemwise_a", 4, 256, 0).fusible(true).finish();
        let c = b.add_kernel("elemwise_c", 4, 256, 0).fusible(true).finish();
        let join = b.add_kernel("combine", 4, 256, 0).fusible(true).finish();
        // r fans out to a and c; both feed join (diamond).
        b.dep(r, a);
        b.dep(r, c);
        b.dep(a, join);
        b.dep(c, join);
        let g = b.build().expect("graph builds");
        let plan = analyse(&g).expect("analysis succeeds");
        assert_eq!(plan.fusion_count(), 1);
        let group = plan.group_of(r).expect("r in a region");
        assert_eq!(group.pattern, ReductionPattern::Generic);
        assert_eq!(group.size(), 4);
    }
}
