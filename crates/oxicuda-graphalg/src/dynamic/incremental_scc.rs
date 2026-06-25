//! Incremental strongly-connected-component maintenance under edge updates.
//!
//! [`IncrementalScc`] tracks the SCC partition of a [`DynamicGraph`] while edges
//! are inserted and deleted one (or a batch) at a time, recomputing only the
//! part of the structure that an update can actually change. The key structural
//! facts exploited are:
//!
//! * **Insertion `u → v` can only *merge* SCCs, never split one.** A merge
//!   happens exactly when `v` could already reach `u` (the new edge then closes
//!   a directed cycle). The SCCs that coalesce are precisely those lying on some
//!   `comp(v) ⇝ comp(u)` path in the *condensation* DAG — i.e. the components
//!   reachable from `comp(v)` **and** co-reachable to `comp(u)`. We find that set
//!   by a forward search from `comp(v)` and a backward search to `comp(u)` over
//!   the condensation and fuse the intersection into one component.
//!   When `v` cannot reach `u`, the partition is unchanged (the common case),
//!   and the update costs only the reachability probe — no global recomputation.
//!
//! * **Deletion `u → v` can only *split* an SCC, never merge.** It can affect at
//!   most the single component `C = comp(u)`, and only when `comp(u) == comp(v)`.
//!   We then re-run Tarjan **restricted to the induced subgraph on `C`** and
//!   replace `C` with the resulting sub-components. A deletion of a
//!   cross-component edge changes nothing.
//!
//! Both paths are exact: after any sequence of updates the maintained partition
//! equals a from-scratch [`scc_tarjan`]
//! recomputation on the current graph (verified in the test suite over randomised
//! update streams). Component identifiers are dense in `0..num_components()` and
//! are renumbered after structural changes.

use std::collections::VecDeque;

use crate::connectivity::scc_tarjan::scc_tarjan;
use crate::dynamic::dynamic_graph::DynamicGraph;
use crate::error::{GraphalgError, GraphalgResult};

/// Incremental SCC tracker over a mutable directed graph.
#[derive(Debug, Clone)]
pub struct IncrementalScc {
    graph: DynamicGraph,
    /// `comp_of[v]` = dense component id of vertex `v` in `0..num_comp`.
    comp_of: Vec<usize>,
    /// Members of each component (`members[c]` is the vertex list of comp `c`).
    members: Vec<Vec<usize>>,
}

impl IncrementalScc {
    /// Build the tracker, computing the initial SCC partition with Tarjan.
    pub fn new(graph: DynamicGraph) -> GraphalgResult<Self> {
        let n = graph.num_nodes();
        let mut me = Self {
            graph,
            comp_of: vec![0; n],
            members: Vec::new(),
        };
        me.full_recompute()?;
        Ok(me)
    }

    /// Number of vertices.
    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    /// Number of strongly-connected components.
    pub fn num_components(&self) -> usize {
        self.members.len()
    }

    /// Dense component id of vertex `v`.
    pub fn component_of(&self, v: usize) -> GraphalgResult<usize> {
        if v >= self.graph.num_nodes() {
            return Err(GraphalgError::IndexOutOfBounds {
                index: v,
                len: self.graph.num_nodes(),
            });
        }
        Ok(self.comp_of[v])
    }

    /// Whether `u` and `v` are in the same SCC (mutually reachable).
    pub fn same_component(&self, u: usize, v: usize) -> GraphalgResult<bool> {
        Ok(self.component_of(u)? == self.component_of(v)?)
    }

    /// The component partition as sorted vertex lists, each list sorted and the
    /// outer list ordered by smallest member.
    pub fn components(&self) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = self
            .members
            .iter()
            .map(|m| {
                let mut v = m.clone();
                v.sort_unstable();
                v
            })
            .collect();
        out.sort_by_key(|c| c.first().copied().unwrap_or(usize::MAX));
        out
    }

    /// Borrow the underlying graph.
    pub fn graph(&self) -> &DynamicGraph {
        &self.graph
    }

    /// Insert directed edge `u → v` and update the partition incrementally.
    pub fn apply_insert(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        self.graph.add_edge(u, v)?;
        self.handle_insert(u, v)
    }

    /// Remove directed edge `u → v` and update the partition incrementally.
    ///
    /// Removing an edge never changes any vertex's *current* component label, so
    /// the affected-component test inside `handle_remove`
    /// can safely read `comp_of` after the graph mutation.
    pub fn apply_remove(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        self.graph.remove_edge(u, v)?;
        self.handle_remove(u, v)
    }

    /// Apply a batch of `(u, v, is_insert)` updates, maintaining the partition
    /// after each one.
    pub fn apply_batch(&mut self, updates: &[(usize, usize, bool)]) -> GraphalgResult<()> {
        for &(u, v, is_insert) in updates {
            if is_insert {
                self.apply_insert(u, v)?;
            } else {
                self.apply_remove(u, v)?;
            }
        }
        Ok(())
    }

    // --- internal machinery -------------------------------------------------

    /// Recompute the whole partition from scratch (used at construction).
    fn full_recompute(&mut self) -> GraphalgResult<()> {
        let g = self.graph.to_adjacency_list();
        let sccs = scc_tarjan(&g)?;
        self.install_partition(sccs);
        Ok(())
    }

    /// Install a fresh list of component vertex-lists, assigning dense ids.
    fn install_partition(&mut self, sccs: Vec<Vec<usize>>) {
        let n = self.graph.num_nodes();
        let mut comp_of = vec![usize::MAX; n];
        let mut members: Vec<Vec<usize>> = Vec::with_capacity(sccs.len());
        for comp in sccs {
            let id = members.len();
            for &v in &comp {
                comp_of[v] = id;
            }
            members.push(comp);
        }
        // Tarjan returns every vertex exactly once, so `comp_of` is fully set.
        self.comp_of = comp_of;
        self.members = members;
    }

    /// Insertion handler: merge the components on a `comp(v) ⇝ comp(u)` cycle.
    fn handle_insert(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        let cu = self.comp_of[u];
        let cv = self.comp_of[v];
        if cu == cv {
            // Endpoints already in the same SCC: nothing changes.
            return Ok(());
        }
        // Components reachable from `cv` in the condensation that can also reach
        // `cu` form the merge set. If `cu` is not forward-reachable from `cv`
        // there is no new cycle and the partition is unchanged.
        let merge = self.condensation_cycle_components(cv, cu);
        if merge.len() <= 1 {
            return Ok(());
        }
        self.merge_components(&merge);
        Ok(())
    }

    /// Deletion handler: re-split `comp(u)` if `u` and `v` shared a component.
    fn handle_remove(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        let n = self.graph.num_nodes();
        if u >= n || v >= n {
            return Ok(());
        }
        let cu = self.comp_of[u];
        let cv = self.comp_of[v];
        if cu != cv {
            // Cross-component edge removal cannot change SCC structure.
            return Ok(());
        }
        // Re-run Tarjan on the induced subgraph of the affected component only.
        let component = self.members[cu].clone();
        let sub = self.tarjan_on_subset(&component)?;
        if sub.len() == 1 {
            // Still one SCC: membership unchanged.
            return Ok(());
        }
        self.replace_component(cu, sub);
        Ok(())
    }

    /// Set of condensation nodes reachable from `start_comp` that can also reach
    /// `target_comp`. Empty/singleton when `target_comp` is not reachable from
    /// `start_comp`. Operates over the *current* partition; condensation edges are
    /// derived on the fly from `graph` + `comp_of`.
    fn condensation_cycle_components(&self, start_comp: usize, target_comp: usize) -> Vec<usize> {
        let nc = self.members.len();
        // Forward reachability from start_comp over condensation edges.
        let forward = self.condensation_reach(start_comp, false);
        if !forward[target_comp] {
            return Vec::new();
        }
        // Backward reachability to target_comp (forward over reversed edges).
        let backward = self.condensation_reach(target_comp, true);
        let mut out = Vec::new();
        for c in 0..nc {
            if forward[c] && backward[c] {
                out.push(c);
            }
        }
        out
    }

    /// BFS over the condensation from `src`. When `reverse` is true, traverse
    /// condensation edges backwards (i.e. predecessors). Self-loops are ignored;
    /// the source is always marked reachable.
    fn condensation_reach(&self, src: usize, reverse: bool) -> Vec<bool> {
        let nc = self.members.len();
        let mut seen = vec![false; nc];
        if src >= nc {
            return seen;
        }
        seen[src] = true;
        let mut queue = VecDeque::new();
        queue.push_back(src);
        while let Some(c) = queue.pop_front() {
            // Enumerate condensation neighbours of component `c`.
            for &x in &self.members[c] {
                let neigh = if reverse {
                    self.graph.in_neighbors(x)
                } else {
                    self.graph.out_neighbors(x)
                };
                if let Ok(list) = neigh {
                    for &y in list {
                        let cy = self.comp_of[y];
                        if cy != c && !seen[cy] {
                            seen[cy] = true;
                            queue.push_back(cy);
                        }
                    }
                }
            }
        }
        seen
    }

    /// Fuse every component id in `merge` into a single component, then renumber
    /// all components densely. `merge` must contain at least two distinct ids.
    fn merge_components(&mut self, merge: &[usize]) {
        let mut keep = vec![true; self.members.len()];
        let mut fused: Vec<usize> = Vec::new();
        for &c in merge {
            keep[c] = false;
            fused.extend(std::mem::take(&mut self.members[c]));
        }
        // Rebuild the member list: surviving components first, then the fused one.
        let mut new_members: Vec<Vec<usize>> = Vec::with_capacity(self.members.len());
        for c in 0..self.members.len() {
            if keep[c] {
                new_members.push(std::mem::take(&mut self.members[c]));
            }
        }
        new_members.push(fused);
        self.reinstall(new_members);
    }

    /// Replace component `cid` with the sub-components `sub` (each a vertex list),
    /// then renumber densely.
    fn replace_component(&mut self, cid: usize, sub: Vec<Vec<usize>>) {
        let mut new_members: Vec<Vec<usize>> = Vec::with_capacity(self.members.len() + sub.len());
        for c in 0..self.members.len() {
            if c == cid {
                continue;
            }
            new_members.push(std::mem::take(&mut self.members[c]));
        }
        for s in sub {
            new_members.push(s);
        }
        self.reinstall(new_members);
    }

    /// Reassign dense component ids from a member list and rebuild `comp_of`.
    fn reinstall(&mut self, members: Vec<Vec<usize>>) {
        let n = self.graph.num_nodes();
        let mut comp_of = vec![usize::MAX; n];
        for (id, comp) in members.iter().enumerate() {
            for &v in comp {
                comp_of[v] = id;
            }
        }
        self.comp_of = comp_of;
        self.members = members;
    }

    /// Run Tarjan on the subgraph induced by `subset` (using only edges with
    /// both endpoints inside the subset), returning the sub-SCCs as vertex lists
    /// in the *original* vertex numbering.
    fn tarjan_on_subset(&self, subset: &[usize]) -> GraphalgResult<Vec<Vec<usize>>> {
        // Map subset vertices to a compact 0..k index space.
        let k = subset.len();
        let mut local = vec![usize::MAX; self.graph.num_nodes()];
        for (i, &v) in subset.iter().enumerate() {
            local[v] = i;
        }
        // Build a small AdjacencyList over the compact space.
        let mut sub = crate::repr::adjacency_list::AdjacencyList::new(k);
        for &v in subset {
            let lv = local[v];
            for &w in self.graph.out_neighbors(v)? {
                let lw = local[w];
                if lw != usize::MAX {
                    // multiplicity is irrelevant to reachability/SCC structure
                    sub.add_edge(lv, lw)?;
                }
            }
        }
        let local_sccs = scc_tarjan(&sub)?;
        // Translate back to original ids.
        let translated = local_sccs
            .into_iter()
            .map(|comp| comp.into_iter().map(|li| subset[li]).collect::<Vec<_>>())
            .collect();
        Ok(translated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Canonicalise a partition (sorted vertex lists, sorted by first member) for
    /// equality comparison.
    fn canon(mut parts: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        for p in &mut parts {
            p.sort_unstable();
        }
        parts.sort_by_key(|p| p.first().copied().unwrap_or(usize::MAX));
        parts
    }

    /// Ground-truth partition via from-scratch Tarjan on the dynamic graph.
    fn ground_truth(g: &DynamicGraph) -> Vec<Vec<usize>> {
        canon(scc_tarjan(&g.to_adjacency_list()).expect("tarjan"))
    }

    #[test]
    fn insertion_merges_into_cycle() {
        // 0 -> 1 -> 2 ; then add 2 -> 0 to form one SCC.
        let mut g = DynamicGraph::new(3);
        g.add_edge(0, 1).expect("e");
        g.add_edge(1, 2).expect("e");
        let mut scc = IncrementalScc::new(g).expect("new");
        assert_eq!(scc.num_components(), 3);
        scc.apply_insert(2, 0).expect("insert");
        assert_eq!(scc.num_components(), 1);
        assert!(scc.same_component(0, 2).expect("same"));
        assert_eq!(scc.components(), ground_truth(scc.graph()));
    }

    #[test]
    fn insertion_without_cycle_keeps_partition() {
        let mut g = DynamicGraph::new(4);
        g.add_edge(0, 1).expect("e");
        g.add_edge(2, 3).expect("e");
        let mut scc = IncrementalScc::new(g).expect("new");
        let before = scc.num_components();
        scc.apply_insert(1, 2).expect("insert"); // bridges but no cycle
        assert_eq!(scc.num_components(), before);
        assert_eq!(scc.components(), ground_truth(scc.graph()));
    }

    #[test]
    fn deletion_splits_cycle() {
        // 0 <-> 1 <-> 2 cycle 0->1->2->0; remove 2->0 splits it.
        let mut g = DynamicGraph::new(3);
        g.add_edge(0, 1).expect("e");
        g.add_edge(1, 2).expect("e");
        g.add_edge(2, 0).expect("e");
        let mut scc = IncrementalScc::new(g).expect("new");
        assert_eq!(scc.num_components(), 1);
        scc.apply_remove(2, 0).expect("remove");
        assert_eq!(scc.num_components(), 3);
        assert_eq!(scc.components(), ground_truth(scc.graph()));
    }

    #[test]
    fn deletion_of_cross_edge_noop() {
        let mut g = DynamicGraph::new(4);
        g.add_edge(0, 1).expect("e");
        g.add_edge(1, 0).expect("e"); // {0,1} SCC
        g.add_edge(1, 2).expect("e"); // cross edge
        g.add_edge(2, 3).expect("e");
        let mut scc = IncrementalScc::new(g).expect("new");
        let before = scc.components();
        scc.apply_remove(1, 2).expect("remove");
        assert_eq!(scc.components(), before);
        assert_eq!(scc.components(), ground_truth(scc.graph()));
    }

    #[test]
    fn deletion_keeps_scc_when_alternate_path() {
        // Triangle 0->1->2->0 plus chord 0->2 means removing 0->1... still need
        // the cycle. Use 0->1->2->0 and a parallel 1->2 so removing one 1->2 keeps
        // the SCC intact (multiplicity).
        let mut g = DynamicGraph::new(3);
        g.add_edge(0, 1).expect("e");
        g.add_edge(1, 2).expect("e");
        g.add_edge(1, 2).expect("e"); // parallel
        g.add_edge(2, 0).expect("e");
        let mut scc = IncrementalScc::new(g).expect("new");
        assert_eq!(scc.num_components(), 1);
        scc.apply_remove(1, 2).expect("remove"); // still one 1->2 left
        assert_eq!(scc.num_components(), 1, "parallel edge keeps SCC");
        assert_eq!(scc.components(), ground_truth(scc.graph()));
    }

    #[test]
    fn merge_of_two_existing_sccs() {
        // Two 2-cycles {0,1} and {2,3}; an edge 1->2 then 3->0 merges all four.
        let mut g = DynamicGraph::new(4);
        g.add_edge(0, 1).expect("e");
        g.add_edge(1, 0).expect("e");
        g.add_edge(2, 3).expect("e");
        g.add_edge(3, 2).expect("e");
        let mut scc = IncrementalScc::new(g).expect("new");
        assert_eq!(scc.num_components(), 2);
        scc.apply_insert(1, 2).expect("insert"); // no cycle yet
        assert_eq!(scc.num_components(), 2);
        scc.apply_insert(3, 0).expect("insert"); // now {0,1,2,3}
        assert_eq!(scc.num_components(), 1);
        assert_eq!(scc.components(), ground_truth(scc.graph()));
    }

    #[test]
    fn random_update_stream_matches_tarjan() {
        let mut rng = LcgRng::new(0x5EED_1234);
        for trial in 0..40 {
            let n = 4 + rng.next_usize(7); // 4..=10
            let mut scc = IncrementalScc::new(DynamicGraph::new(n)).expect("new");
            // Track present distinct edges to issue legal removals.
            let mut present: Vec<(usize, usize)> = Vec::new();
            for step in 0..60 {
                let do_insert = present.is_empty() || rng.next_bool();
                if do_insert {
                    let u = rng.next_usize(n);
                    let mut v = rng.next_usize(n);
                    if v == u {
                        v = (v + 1) % n;
                    }
                    scc.apply_insert(u, v).expect("insert");
                    if !present.contains(&(u, v)) {
                        present.push((u, v));
                    }
                } else {
                    let idx = rng.next_usize(present.len());
                    let (u, v) = present[idx];
                    scc.apply_remove(u, v).expect("remove");
                    // multiplicity is always one in this stream, so it's gone
                    present.swap_remove(idx);
                }
                assert_eq!(
                    scc.components(),
                    ground_truth(scc.graph()),
                    "mismatch trial {trial} step {step} (n={n})"
                );
                // comp_of must be consistent with components()
                for v in 0..n {
                    let cid = scc.component_of(v).expect("comp");
                    assert!(scc.members[cid].contains(&v));
                }
            }
        }
    }

    #[test]
    fn batch_update_matches_tarjan() {
        let mut g = DynamicGraph::new(5);
        g.add_edge(0, 1).expect("e");
        g.add_edge(1, 2).expect("e");
        let mut scc = IncrementalScc::new(g).expect("new");
        let updates = [
            (2, 0, true),  // form {0,1,2}
            (3, 4, true),  //
            (4, 3, true),  // form {3,4}
            (2, 3, true),  // cross, no merge
            (0, 1, false), // remove one of the cycle edges
        ];
        scc.apply_batch(&updates).expect("batch");
        assert_eq!(scc.components(), ground_truth(scc.graph()));
    }

    #[test]
    fn self_loop_is_its_own_scc_member() {
        let mut g = DynamicGraph::new(2);
        g.add_edge(0, 0).expect("e"); // self-loop
        g.add_edge(0, 1).expect("e");
        let scc = IncrementalScc::new(g).expect("new");
        assert_eq!(scc.num_components(), 2);
        assert_eq!(scc.components(), ground_truth(scc.graph()));
    }

    #[test]
    fn component_of_out_of_range_errors() {
        let scc = IncrementalScc::new(DynamicGraph::new(2)).expect("new");
        assert!(scc.component_of(5).is_err());
    }
}
