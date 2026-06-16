//! Lipton-Tarjan style planar separator (1979).
//!
//! Given a (planar) graph on `n` vertices, computes a vertex separator `S`
//! together with a partition of the remaining vertices into `A` and `B` such
//! that
//!
//! * there is **no edge** between `A` and `B` (removing `S` disconnects them),
//! * each side is balanced: `|A| <= 2n/3` and `|B| <= 2n/3`.
//!
//! # Construction (scope)
//!
//! This is a *compact* realisation of the Lipton-Tarjan program, built from the
//! two ingredients the theorem relies on, chosen per input:
//!
//! * **BFS levelling** (the heart of Lipton-Tarjan). Run a breadth-first search
//!   from a root. Because every edge of a BFS-layered graph joins vertices in
//!   the same or adjacent levels, deleting *one entire level* `l` separates the
//!   strictly-lower levels from the strictly-higher levels. The *median* level
//!   always balances both sides below `n/2 <= 2n/3`; among all levels that keep
//!   both sides within `2n/3` we pick the **smallest** one as the separator. On
//!   a `sqrt(n) x sqrt(n)` grid the BFS levels are anti-diagonals of size
//!   `O(sqrt(n))`, giving the textbook `O(sqrt(n))` separator.
//!
//! * **Tree centroid** specialisation. For a tree (the degenerate planar case)
//!   the BFS-level separator can be as large as `n/2`, yet a single *centroid*
//!   vertex already splits every branch below `n/2`. When the input component is
//!   a tree we therefore return the centroid (a size-`1` separator) and balance
//!   its branches into `A`/`B` with an exact subset-sum partition.
//!
//! Disconnected inputs are handled component-wise. The balance and
//! "no `A`-`B` edge" guarantees hold for every input; the `O(sqrt(n))` *size*
//! guarantee is for the planar classes exercised in the tests (grids, trees,
//! paths), matching the latitude of the assignment.

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

/// The three parts of a separator decomposition. Together they partition
/// `0..n` with no edge crossing between [`SeparatorResult::part_a`] and
/// [`SeparatorResult::part_b`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeparatorResult {
    /// First balanced part.
    pub part_a: Vec<usize>,
    /// The separator (whose removal disconnects `A` from `B`).
    pub separator: Vec<usize>,
    /// Second balanced part.
    pub part_b: Vec<usize>,
}

/// Planar separator solver over an undirected graph.
#[derive(Debug, Clone)]
pub struct PlanarSeparator {
    n: usize,
    adj: Vec<Vec<usize>>,
}

impl PlanarSeparator {
    /// Create an edgeless solver on `n` vertices.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
        }
    }

    /// Add an undirected edge `(u, v)`. Self-loops are ignored; parallel edges
    /// are de-duplicated.
    pub fn add_edge(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        if u >= self.n || v >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u.max(v),
                len: self.n,
            });
        }
        if u == v {
            return Ok(());
        }
        if !self.adj[u].contains(&v) {
            self.adj[u].push(v);
        }
        if !self.adj[v].contains(&u) {
            self.adj[v].push(u);
        }
        Ok(())
    }

    /// Build a planar separator solver from an [`AdjacencyList`], symmetrising
    /// the edges (the graph is treated as undirected).
    pub fn from_adjacency_list(g: &AdjacencyList) -> GraphalgResult<Self> {
        let mut sep = Self::new(g.n);
        for u in 0..g.n {
            for &v in &g.edges[u] {
                sep.add_edge(u, v)?;
            }
        }
        Ok(sep)
    }

    /// Number of vertices.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Compute a balanced vertex separator.
    pub fn separate(&self) -> GraphalgResult<SeparatorResult> {
        if self.n == 0 {
            return Ok(SeparatorResult {
                part_a: Vec::new(),
                separator: Vec::new(),
                part_b: Vec::new(),
            });
        }
        if self.n == 1 {
            return Ok(SeparatorResult {
                part_a: vec![0],
                separator: Vec::new(),
                part_b: Vec::new(),
            });
        }
        let comps = self.connected_components();
        let limit = (2 * self.n) / 3;
        if comps.len() == 1 {
            let (a, s, b) = self.separate_component(&comps[0])?;
            return Ok(finalize(a, s, b));
        }
        // Multiple components.
        let max_comp = comps.iter().map(|c| c.len()).max().unwrap_or(0);
        if max_comp <= limit {
            // No separator needed: bin-pack components into A and B.
            let sizes: Vec<usize> = comps.iter().map(|c| c.len()).collect();
            let (a_idx, b_idx) = best_partition(&sizes);
            let mut part_a = Vec::new();
            let mut part_b = Vec::new();
            for &i in &a_idx {
                part_a.extend_from_slice(&comps[i]);
            }
            for &i in &b_idx {
                part_b.extend_from_slice(&comps[i]);
            }
            return Ok(finalize(part_a, Vec::new(), part_b));
        }
        // One giant component dominates: separate inside it and distribute the
        // remaining (small) components onto the lighter side.
        let giant_idx = comps
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| c.len())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let (mut ga, gs, mut gb) = self.separate_component(&comps[giant_idx])?;
        for (i, c) in comps.iter().enumerate() {
            if i == giant_idx {
                continue;
            }
            if ga.len() <= gb.len() {
                ga.extend_from_slice(c);
            } else {
                gb.extend_from_slice(c);
            }
        }
        Ok(finalize(ga, gs, gb))
    }

    /// Connected components as vertex lists.
    fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut comp_id = vec![usize::MAX; self.n];
        let mut comps: Vec<Vec<usize>> = Vec::new();
        for start in 0..self.n {
            if comp_id[start] != usize::MAX {
                continue;
            }
            let id = comps.len();
            let mut members = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            comp_id[start] = id;
            while let Some(u) = queue.pop_front() {
                members.push(u);
                for &w in &self.adj[u] {
                    if comp_id[w] == usize::MAX {
                        comp_id[w] = id;
                        queue.push_back(w);
                    }
                }
            }
            comps.push(members);
        }
        comps
    }

    /// Number of undirected edges with both endpoints inside `verts`.
    fn component_edge_count(&self, in_set: &[bool], verts: &[usize]) -> usize {
        let mut deg_sum = 0usize;
        for &u in verts {
            for &w in &self.adj[u] {
                if in_set[w] {
                    deg_sum += 1;
                }
            }
        }
        deg_sum / 2
    }

    /// Separate a single connected component (`verts`) into `(A, S, B)`.
    fn separate_component(
        &self,
        verts: &[usize],
    ) -> GraphalgResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let k = verts.len();
        if k == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        if k == 1 {
            return Ok((vec![verts[0]], Vec::new(), Vec::new()));
        }
        let mut in_set = vec![false; self.n];
        for &v in verts {
            in_set[v] = true;
        }
        let edge_count = self.component_edge_count(&in_set, verts);
        if edge_count == k - 1 {
            self.tree_centroid_separator(verts, &in_set)
        } else {
            self.bfs_level_separator(verts, &in_set)
        }
    }

    /// Tree case: remove a centroid, split its branches into `A`/`B`.
    fn tree_centroid_separator(
        &self,
        verts: &[usize],
        in_set: &[bool],
    ) -> GraphalgResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let k = verts.len();
        let root = verts[0];
        // Rooted BFS order + parents.
        let mut parent = vec![usize::MAX; self.n];
        let mut visited = vec![false; self.n];
        let mut order = Vec::with_capacity(k);
        let mut queue = VecDeque::new();
        queue.push_back(root);
        visited[root] = true;
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &w in &self.adj[u] {
                if in_set[w] && !visited[w] {
                    visited[w] = true;
                    parent[w] = u;
                    queue.push_back(w);
                }
            }
        }
        // Subtree sizes (reverse BFS order).
        let mut size = vec![0usize; self.n];
        for &v in verts {
            size[v] = 1;
        }
        for &u in order.iter().rev() {
            if parent[u] != usize::MAX {
                size[parent[u]] += size[u];
            }
        }
        // Centroid: minimise the largest component after removal.
        let mut best_c = root;
        let mut best_max = usize::MAX;
        for &u in verts {
            let mut mx = k - size[u]; // parent side
            for &w in &self.adj[u] {
                if in_set[w] && parent[w] == u {
                    mx = mx.max(size[w]);
                }
            }
            if mx < best_max {
                best_max = mx;
                best_c = u;
            }
        }
        // Branch components after removing best_c.
        let mut removed = vec![false; self.n];
        removed[best_c] = true;
        let mut branch_visited = vec![false; self.n];
        branch_visited[best_c] = true;
        let mut branches: Vec<Vec<usize>> = Vec::new();
        for &w in &self.adj[best_c] {
            if in_set[w] && !branch_visited[w] {
                let mut comp = Vec::new();
                let mut q = VecDeque::new();
                q.push_back(w);
                branch_visited[w] = true;
                while let Some(u) = q.pop_front() {
                    comp.push(u);
                    for &x in &self.adj[u] {
                        if in_set[x] && !branch_visited[x] && x != best_c {
                            branch_visited[x] = true;
                            q.push_back(x);
                        }
                    }
                }
                branches.push(comp);
            }
        }
        let sizes: Vec<usize> = branches.iter().map(|c| c.len()).collect();
        let (a_idx, b_idx) = best_partition(&sizes);
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();
        for &i in &a_idx {
            part_a.extend_from_slice(&branches[i]);
        }
        for &i in &b_idx {
            part_b.extend_from_slice(&branches[i]);
        }
        Ok((part_a, vec![best_c], part_b))
    }

    /// General case: separate by removing the smallest balancing BFS level.
    fn bfs_level_separator(
        &self,
        verts: &[usize],
        in_set: &[bool],
    ) -> GraphalgResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let k = verts.len();
        let root = verts[0];
        let mut level = vec![usize::MAX; self.n];
        let mut queue = VecDeque::new();
        level[root] = 0;
        queue.push_back(root);
        let mut max_level = 0usize;
        while let Some(u) = queue.pop_front() {
            let lu = level[u];
            if lu > max_level {
                max_level = lu;
            }
            for &w in &self.adj[u] {
                if in_set[w] && level[w] == usize::MAX {
                    level[w] = lu + 1;
                    queue.push_back(w);
                }
            }
        }
        // Level sizes and cumulative counts.
        let mut level_size = vec![0usize; max_level + 1];
        for &v in verts {
            level_size[level[v]] += 1;
        }
        let mut prefix = vec![0usize; max_level + 2];
        for l in 0..=max_level {
            prefix[l + 1] = prefix[l] + level_size[l];
        }
        let limit = (2 * k) / 3;
        // Among levels keeping both sides within 2k/3, pick the smallest level.
        let mut chosen = usize::MAX;
        let mut chosen_size = usize::MAX;
        for l in 0..=max_level {
            let below = prefix[l];
            let above = k - prefix[l + 1];
            if below <= limit && above <= limit && level_size[l] < chosen_size {
                chosen_size = level_size[l];
                chosen = l;
            }
        }
        // The median level always qualifies, so `chosen` is set; guard anyway.
        if chosen == usize::MAX {
            chosen = 0;
        }
        let mut part_a = Vec::new();
        let mut separator = Vec::new();
        let mut part_b = Vec::new();
        for &v in verts {
            let lv = level[v];
            if lv < chosen {
                part_a.push(v);
            } else if lv == chosen {
                separator.push(v);
            } else {
                part_b.push(v);
            }
        }
        Ok((part_a, separator, part_b))
    }
}

/// Sort each part and assemble a [`SeparatorResult`].
fn finalize(mut a: Vec<usize>, mut s: Vec<usize>, mut b: Vec<usize>) -> SeparatorResult {
    a.sort_unstable();
    s.sort_unstable();
    b.sort_unstable();
    SeparatorResult {
        part_a: a,
        separator: s,
        part_b: b,
    }
}

/// Split item `sizes` into two index groups whose total sizes are as balanced
/// as possible, via exact subset-sum dynamic programming.
fn best_partition(sizes: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let m = sizes.len();
    if m == 0 {
        return (Vec::new(), Vec::new());
    }
    let total: usize = sizes.iter().sum();
    if total == 0 {
        return ((0..m).collect(), Vec::new());
    }
    let mut reach = vec![vec![false; total + 1]; m + 1];
    reach[0][0] = true;
    for i in 1..=m {
        let s = sizes[i - 1];
        for x in 0..=total {
            if reach[i - 1][x] {
                reach[i][x] = true;
                if x + s <= total {
                    reach[i][x + s] = true;
                }
            }
        }
    }
    let target = total / 2;
    let mut best_x = 0usize;
    let mut best_diff = usize::MAX;
    for x in 0..=total {
        if reach[m][x] {
            let diff = x.abs_diff(target);
            if diff < best_diff {
                best_diff = diff;
                best_x = x;
            }
        }
    }
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut x = best_x;
    for i in (1..=m).rev() {
        let s = sizes[i - 1];
        if x >= s && reach[i - 1][x - s] {
            a.push(i - 1);
            x -= s;
        } else {
            b.push(i - 1);
        }
    }
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Verify the structural separator invariants against an explicit edge list.
    fn check_separator(
        n: usize,
        edges: &[(usize, usize)],
        res: &SeparatorResult,
        balance_limit: usize,
    ) {
        // Partition of 0..n with no overlaps.
        let mut seen = vec![0u8; n];
        for &v in res.part_a.iter().chain(&res.separator).chain(&res.part_b) {
            seen[v] += 1;
        }
        for (v, &c) in seen.iter().enumerate() {
            assert_eq!(c, 1, "vertex {v} appears {c} times (must be exactly once)");
        }
        // No edge between A and B.
        let a: BTreeSet<usize> = res.part_a.iter().copied().collect();
        let b: BTreeSet<usize> = res.part_b.iter().copied().collect();
        for &(u, v) in edges {
            let ab = a.contains(&u) && b.contains(&v);
            let ba = b.contains(&u) && a.contains(&v);
            assert!(!ab && !ba, "edge ({u},{v}) crosses A-B separator");
        }
        // Balance.
        assert!(
            res.part_a.len() <= balance_limit,
            "|A|={} exceeds {balance_limit}",
            res.part_a.len()
        );
        assert!(
            res.part_b.len() <= balance_limit,
            "|B|={} exceeds {balance_limit}",
            res.part_b.len()
        );
    }

    fn build_grid(a: usize) -> (PlanarSeparator, Vec<(usize, usize)>) {
        let n = a * a;
        let mut sep = PlanarSeparator::new(n);
        let mut edges = Vec::new();
        for i in 0..a {
            for j in 0..a {
                let id = i * a + j;
                if j + 1 < a {
                    sep.add_edge(id, id + 1).expect("edge");
                    edges.push((id, id + 1));
                }
                if i + 1 < a {
                    sep.add_edge(id, id + a).expect("edge");
                    edges.push((id, id + a));
                }
            }
        }
        (sep, edges)
    }

    #[test]
    fn grid_separator_is_small_and_balanced() {
        let a = 7;
        let n = a * a;
        let (sep, edges) = build_grid(a);
        let res = sep.separate().expect("separate");
        let limit = (2 * n) / 3;
        check_separator(n, &edges, &res, limit);
        // O(sqrt(n)): a single anti-diagonal/row is ~ sqrt(n).
        let bound = (3.0 * (n as f64).sqrt()).ceil() as usize;
        assert!(
            res.separator.len() <= bound,
            "separator size {} exceeds O(sqrt(n)) bound {bound}",
            res.separator.len()
        );
        // Removing S must leave both sides non-empty for a grid (real split).
        assert!(!res.part_a.is_empty() && !res.part_b.is_empty());
    }

    #[test]
    fn grid_partition_property_holds() {
        let a = 5;
        let n = a * a;
        let (sep, edges) = build_grid(a);
        let res = sep.separate().expect("separate");
        check_separator(n, &edges, &res, (2 * n) / 3);
    }

    #[test]
    fn path_has_size_one_separator() {
        let n = 11;
        let mut sep = PlanarSeparator::new(n);
        let mut edges = Vec::new();
        for i in 0..n - 1 {
            sep.add_edge(i, i + 1).expect("edge");
            edges.push((i, i + 1));
        }
        let res = sep.separate().expect("separate");
        check_separator(n, &edges, &res, (2 * n) / 3);
        assert_eq!(
            res.separator.len(),
            1,
            "path centroid separator is one vertex"
        );
    }

    #[test]
    fn balanced_binary_tree_uses_centroid() {
        // Complete binary tree with 15 nodes: BFS-level would remove ~8 leaves,
        // but the centroid is a single vertex.
        let n = 15;
        let mut sep = PlanarSeparator::new(n);
        let mut edges = Vec::new();
        for i in 1..n {
            let p = (i - 1) / 2;
            sep.add_edge(p, i).expect("edge");
            edges.push((p, i));
        }
        let res = sep.separate().expect("separate");
        check_separator(n, &edges, &res, (2 * n) / 3);
        assert_eq!(res.separator.len(), 1, "tree separator is the centroid");
    }

    #[test]
    fn star_separator_is_center() {
        let n = 9;
        let mut sep = PlanarSeparator::new(n);
        let mut edges = Vec::new();
        for i in 1..n {
            sep.add_edge(0, i).expect("edge");
            edges.push((0, i));
        }
        let res = sep.separate().expect("separate");
        check_separator(n, &edges, &res, (2 * n) / 3);
        // Removing the hub disconnects everything; centroid = hub.
        assert_eq!(res.separator, vec![0]);
    }

    #[test]
    fn disconnected_components_partition_with_empty_separator() {
        // Two triangles (disjoint), each 3 vertices.
        let n = 6;
        let mut sep = PlanarSeparator::new(n);
        let edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        for &(u, v) in &edges {
            sep.add_edge(u, v).expect("edge");
        }
        let res = sep.separate().expect("separate");
        check_separator(n, &edges, &res, (2 * n) / 3);
        assert!(
            res.separator.is_empty(),
            "two equal triangles split with no separator"
        );
        assert_eq!(res.part_a.len(), 3);
        assert_eq!(res.part_b.len(), 3);
    }

    #[test]
    fn trivial_graphs_handled() {
        let empty = PlanarSeparator::new(0);
        let r0 = empty.separate().expect("separate");
        assert!(r0.part_a.is_empty() && r0.separator.is_empty() && r0.part_b.is_empty());

        let single = PlanarSeparator::new(1);
        let r1 = single.separate().expect("separate");
        assert_eq!(r1.part_a, vec![0]);
        assert!(r1.separator.is_empty() && r1.part_b.is_empty());
    }

    #[test]
    fn rejects_out_of_range_edge() {
        let mut sep = PlanarSeparator::new(3);
        assert!(sep.add_edge(0, 5).is_err());
    }

    #[test]
    fn from_adjacency_list_round_trip() {
        let mut g = AdjacencyList::new(4);
        g.add_undirected_edge(0, 1).expect("edge");
        g.add_undirected_edge(1, 2).expect("edge");
        g.add_undirected_edge(2, 3).expect("edge");
        let sep = PlanarSeparator::from_adjacency_list(&g).expect("build");
        let res = sep.separate().expect("separate");
        let edges = [(0, 1), (1, 2), (2, 3)];
        check_separator(4, &edges, &res, (2 * 4) / 3);
    }
}
