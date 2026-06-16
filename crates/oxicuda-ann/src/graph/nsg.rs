//! NSG — Navigating Spreading-out Graph.
//!
//! Reference: Cong Fu, Chao Xiang, Changxu Wang, Deng Cai, *"Fast Approximate
//! Nearest Neighbor Search With The Navigating Spreading-out Graph"*, PVLDB
//! 12(5):461–474, 2019.
//!
//! NSG is a monotonic, navigable proximity graph derived from an *approximate*
//! k-NN graph. Its edge-selection rule is the **MRNG (Monotonic Relative
//! Neighborhood Graph)** occlusion test, which yields a sparse graph with a
//! bounded out-degree while preserving a *monotonic search path* from a single
//! **navigating node** to (almost) every other node.
//!
//! # Build outline (paraphrased from the paper)
//!
//! 1. Build an approximate k-NN graph over the data (here: NN-Descent, reusing
//!    [`crate::knn_graph::knn_graph::KnnGraph`]).
//! 2. Choose the **navigating node**: the medoid, i.e. the data point closest
//!    to the global centroid. Every query search is seeded from it.
//! 3. For every node `p`, gather a candidate pool `C(p)` — the union of (i) the
//!    greedy-search visited set from the navigating node towards `p` over the
//!    k-NN graph and (ii) `p`'s own approximate k-NN neighbours — then sort it
//!    ascending by `dist(p, ·)`.
//! 4. Apply the **MRNG occlusion rule** to `C(p)` to pick at most `R`
//!    out-neighbours: accept the closest candidate `q`; thereafter a candidate
//!    `t` is accepted only if it is **not occluded** by any already-accepted
//!    neighbour `r`, i.e. there is no `r` with `dist(r, t) < dist(p, t)`
//!    (equivalently `t` lies outside every "lune" of accepted edges).
//! 5. Add the chosen edges. NSG is built as a (mostly) directed graph; we add
//!    the back-edge `q → p` opportunistically so the navigating node can reach
//!    the whole graph, re-applying the occlusion rule (with degree cap `R`) when
//!    a node overflows.
//! 6. **Tree augmentation for connectivity**: run a DFS from the navigating
//!    node. Any node not reached is connected by adding an edge from its nearest
//!    *reached* node (found by a greedy search), guaranteeing the final graph is
//!    connected from the navigating node.
//!
//! # Search
//!
//! Queries are answered by a greedy best-first beam search (search-list size
//! `L`) seeded from the navigating node, identical in spirit to the DiskANN /
//! HNSW layer-0 search. The result is the `k` closest visited nodes, ascending
//! by squared-L2 distance.
//!
//! All ordering uses **squared** Euclidean distance (L2²) via
//! [`crate::distance::l2::l2_sq`]; results are `(u32, f32)` = `(id, dist²)`.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet, VecDeque};

use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::knn_graph::knn_graph::KnnGraph;

/// Totally-ordered `f32` wrapper for binary heaps. `NaN` sinks to the "largest"
/// bucket so it never poisons ordering.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.0.partial_cmp(&other.0) {
            Some(o) => o,
            None => match (self.0.is_nan(), other.0.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => std::cmp::Ordering::Equal,
            },
        }
    }
}

/// NSG build / search configuration.
#[derive(Debug, Clone)]
pub struct NsgConfig {
    /// Maximum out-degree `R` of the final NSG. Must be `>= 1`.
    pub degree_r: usize,
    /// Candidate-pool / search-list size `L` used during construction and as the
    /// default for queries. Must be `>= 1`.
    pub search_l: usize,
    /// Number of neighbours `k` in the seed approximate k-NN graph. Must be
    /// `>= 1`.
    pub knn_k: usize,
    /// NN-Descent iteration budget when building the seed k-NN graph.
    pub nndescent_iters: usize,
}

impl Default for NsgConfig {
    fn default() -> Self {
        Self {
            degree_r: 16,
            search_l: 40,
            knn_k: 16,
            nndescent_iters: 12,
        }
    }
}

/// A built Navigating Spreading-out Graph.
pub struct NsgIndex {
    /// Flat `n × dim` row-major point storage.
    points: Vec<f32>,
    /// `graph[p]` is the bounded out-neighbour list of node `p`.
    graph: Vec<Vec<u32>>,
    /// The navigating node (medoid) seeding every search.
    navigating_node: u32,
    /// Vector dimensionality.
    dim: usize,
    /// Cached configuration.
    cfg: NsgConfig,
}

impl NsgIndex {
    /// Number of indexed points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// `true` when no points are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The navigating node id (medoid) seeding every search.
    #[must_use]
    pub fn navigating_node(&self) -> u32 {
        self.navigating_node
    }

    /// Read-only access to a node's out-neighbour list.
    #[must_use]
    pub fn neighbors(&self, id: u32) -> &[u32] {
        let i = id as usize;
        if i < self.graph.len() {
            &self.graph[i]
        } else {
            &[]
        }
    }

    /// Borrow point `id` (caller guarantees `id < n`).
    fn point(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.points[s..s + self.dim]
    }

    /// Squared-L2 distance between `query` and node `id`.
    fn dist_query(&self, query: &[f32], id: u32) -> f32 {
        let v = self.point(id);
        query
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// Squared-L2 distance between two stored nodes.
    fn dist_nodes(&self, a: u32, b: u32) -> f32 {
        let va = self.point(a);
        let vb = self.point(b);
        va.iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    /// Compute the medoid: the data point closest to the global centroid.
    fn compute_medoid(data: &[f32], n: usize, dim: usize) -> u32 {
        // Global centroid.
        let mut centroid = vec![0.0_f32; dim];
        for row in data.chunks_exact(dim) {
            for (c, &x) in centroid.iter_mut().zip(row.iter()) {
                *c += x;
            }
        }
        let inv = 1.0 / n as f32;
        for c in centroid.iter_mut() {
            *c *= inv;
        }
        // Nearest point to the centroid.
        let mut best = 0u32;
        let mut best_d = f32::INFINITY;
        for (i, row) in data.chunks_exact(dim).enumerate() {
            let d: f32 = row
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d < best_d {
                best_d = d;
                best = i as u32;
            }
        }
        best
    }

    /// Greedy best-first search over `self.graph`, seeded from `start`,
    /// returning the visited set ascending by distance to `query`, bounded to
    /// `l`. `exclude` is never collected nor expanded.
    fn greedy_visit_graph(
        &self,
        query: &[f32],
        start: u32,
        l: usize,
        exclude: Option<u32>,
    ) -> Vec<(u32, f32)> {
        self.greedy_visit_adj(query, start, l, exclude, &self.graph)
    }

    /// Greedy best-first search over an explicit adjacency table `adj`.
    fn greedy_visit_adj(
        &self,
        query: &[f32],
        start: u32,
        l: usize,
        exclude: Option<u32>,
        adj: &[Vec<u32>],
    ) -> Vec<(u32, f32)> {
        let n = adj.len();
        if n == 0 || l == 0 {
            return Vec::new();
        }
        let mut frontier: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
        let mut result: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
        let mut visited: HashSet<u32> = HashSet::new();
        let mut consumed: HashSet<u32> = HashSet::new();

        // If the seed itself is excluded, fall back to any other present id.
        let mut seed = start;
        if Some(seed) == exclude {
            let mut s = 0u32;
            while (s as usize) < n && Some(s) == exclude {
                s += 1;
            }
            if s as usize >= n {
                return Vec::new();
            }
            seed = s;
        }

        let d_seed = self.dist_query(query, seed);
        frontier.push(Reverse((OrdF32(d_seed), seed)));
        result.push((OrdF32(d_seed), seed));
        visited.insert(seed);

        while let Some(Reverse((OrdF32(c_dist), c_id))) = frontier.pop() {
            if !consumed.insert(c_id) {
                continue;
            }
            if result.len() >= l {
                let kth = result.peek().map_or(f32::INFINITY, |(OrdF32(d), _)| *d);
                if c_dist > kth {
                    break;
                }
            }
            for &nbr in &adj[c_id as usize] {
                if Some(nbr) == exclude {
                    continue;
                }
                if !visited.insert(nbr) {
                    continue;
                }
                let d = self.dist_query(query, nbr);
                let worst = result.peek().map_or(f32::INFINITY, |(OrdF32(w), _)| *w);
                if result.len() < l {
                    result.push((OrdF32(d), nbr));
                    frontier.push(Reverse((OrdF32(d), nbr)));
                } else if d < worst {
                    result.pop();
                    result.push((OrdF32(d), nbr));
                    frontier.push(Reverse((OrdF32(d), nbr)));
                }
            }
        }

        let mut out: Vec<(u32, f32)> = result.into_iter().map(|(OrdF32(d), id)| (id, d)).collect();
        out.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        if out.len() > l {
            out.truncate(l);
        }
        out
    }

    /// MRNG occlusion pruning. Given a candidate pool for `p` already sorted
    /// ascending by `dist(p, ·)`, select at most `R` out-neighbours: accept a
    /// candidate `t` only if no already-accepted neighbour `r` is **strictly
    /// closer to `t` than `p` is** (`dist(r, t) < dist(p, t)`). That is, `t` is
    /// dropped when it is *occluded* by an accepted edge.
    ///
    /// `candidates` are `(id, dist_p)` pairs (distance from `p`). Returns the
    /// accepted neighbour ids.
    fn mrng_prune(&self, p: u32, candidates: &[(u32, f32)], r: usize) -> Vec<u32> {
        let mut accepted: Vec<u32> = Vec::with_capacity(r);
        for &(t, d_p_t) in candidates {
            if t == p {
                continue;
            }
            if accepted.len() >= r {
                break;
            }
            // Occlusion test against every already-accepted neighbour.
            let mut occluded = false;
            for &rr in &accepted {
                let d_r_t = self.dist_nodes(rr, t);
                if d_r_t < d_p_t {
                    occluded = true;
                    break;
                }
            }
            if !occluded {
                accepted.push(t);
            }
        }
        accepted
    }

    /// Build an NSG from `data` (row-major `n × dim`).
    ///
    /// # Errors
    /// - [`AnnError::EmptyInput`] if `n == 0`.
    /// - [`AnnError::InvalidVectorDim`] if `dim == 0`.
    /// - [`AnnError::DimensionMismatch`] if `data.len() != n * dim`.
    /// - [`AnnError::Internal`] if `cfg.degree_r`, `cfg.search_l` or `cfg.knn_k`
    ///   is `0`.
    pub fn build(
        data: &[f32],
        n: usize,
        dim: usize,
        cfg: NsgConfig,
        rng: &mut LcgRng,
    ) -> AnnResult<Self> {
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim });
        }
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        if cfg.degree_r == 0 || cfg.search_l == 0 || cfg.knn_k == 0 {
            return Err(AnnError::Internal {
                msg: "degree_r, search_l and knn_k must all be >= 1".to_string(),
            });
        }

        let mut index = Self {
            points: data.to_vec(),
            graph: vec![Vec::new(); n],
            navigating_node: 0,
            dim,
            cfg: cfg.clone(),
        };

        // Trivial single-point graph.
        if n == 1 {
            index.navigating_node = 0;
            return Ok(index);
        }

        // 1. Seed approximate k-NN graph (NN-Descent).
        let knn =
            KnnGraph::build_nn_descent(data, n, dim, cfg.knn_k, cfg.nndescent_iters, 0.001, rng);
        // Materialise the k-NN adjacency as a `Vec<Vec<u32>>` for greedy search.
        let knn_adj: Vec<Vec<u32>> = (0..n)
            .map(|i| knn.neighbors(i).iter().map(|&(id, _)| id).collect())
            .collect();

        // 2. Navigating node = medoid.
        index.navigating_node = Self::compute_medoid(data, n, dim);
        let nav = index.navigating_node;

        // 3 + 4. Per-node candidate pool + MRNG pruning.
        let r = cfg.degree_r;
        let l = cfg.search_l.max(r);
        for p in 0..n as u32 {
            let q_vec: Vec<f32> = index.point(p).to_vec();

            // Candidate pool: greedy-search visited set from the navigating node
            // over the k-NN graph, towards p.
            let mut pool = index.greedy_visit_adj(&q_vec, nav, l, Some(p), &knn_adj);

            // Union p's own approximate k-NN neighbours (these are the closest
            // local candidates and stabilise the pool on clustered data).
            let mut seen: HashSet<u32> = pool.iter().map(|&(id, _)| id).collect();
            for &nbr_id in &knn_adj[p as usize] {
                if nbr_id != p && seen.insert(nbr_id) {
                    let d = index.dist_nodes(p, nbr_id);
                    pool.push((nbr_id, d));
                }
            }

            // Sort ascending by distance to p, then apply MRNG occlusion.
            pool.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            let chosen = index.mrng_prune(p, &pool, r);
            index.graph[p as usize] = chosen;
        }

        // 5. Opportunistic back-edges (so the navigating node can reach the
        // whole graph), re-pruning with MRNG when a node overflows R.
        for p in 0..n as u32 {
            let nbrs = index.graph[p as usize].clone();
            for q in nbrs {
                let qi = q as usize;
                if q != p && !index.graph[qi].contains(&p) {
                    index.graph[qi].push(p);
                    if index.graph[qi].len() > r {
                        // Re-prune q from its current out-neighbours ∪ {p}.
                        let mut cand: Vec<(u32, f32)> = index.graph[qi]
                            .iter()
                            .filter(|&&v| v != q)
                            .map(|&v| (v, index.dist_nodes(q, v)))
                            .collect();
                        cand.sort_unstable_by(|a, b| {
                            a.1.partial_cmp(&b.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then(a.0.cmp(&b.0))
                        });
                        index.graph[qi] = index.mrng_prune(q, &cand, r);
                    }
                }
            }
        }

        // 6. Tree augmentation: guarantee connectivity from the navigating node.
        index.ensure_connectivity(n)?;

        Ok(index)
    }

    /// DFS-based tree augmentation. Any node unreachable from the navigating
    /// node is connected to its nearest already-reached node (found by greedy
    /// search), repeating until every node is reachable.
    fn ensure_connectivity(&mut self, n: usize) -> AnnResult<()> {
        let nav = self.navigating_node;
        loop {
            // BFS/DFS reachability from the navigating node.
            let mut reached = vec![false; n];
            let mut stack: Vec<u32> = vec![nav];
            reached[nav as usize] = true;
            while let Some(u) = stack.pop() {
                for &v in &self.graph[u as usize] {
                    if !reached[v as usize] {
                        reached[v as usize] = true;
                        stack.push(v);
                    }
                }
            }

            // Find the first unreached node.
            let unreached = (0..n).find(|&i| !reached[i]);
            let Some(target) = unreached else {
                break; // fully connected
            };
            let target = target as u32;

            // Greedy search (over the current NSG) from the navigating node to
            // find the closest *reached* node; connect it → target.
            let q_vec: Vec<f32> = self.point(target).to_vec();
            let visited = self.greedy_visit_graph(&q_vec, nav, self.cfg.search_l.max(1), None);
            let mut anchor: Option<u32> = visited
                .iter()
                .find(|&&(id, _)| reached[id as usize] && id != target)
                .map(|&(id, _)| id);
            // Fallback: nearest reached node by brute force (always exists since
            // the navigating node is reached).
            if anchor.is_none() {
                let mut best = nav;
                let mut best_d = f32::INFINITY;
                for (i, &rch) in reached.iter().enumerate() {
                    if rch && i as u32 != target {
                        let d = self.dist_nodes(i as u32, target);
                        if d < best_d {
                            best_d = d;
                            best = i as u32;
                        }
                    }
                }
                anchor = Some(best);
            }
            let anchor = anchor.unwrap_or(nav);

            // Add edge anchor → target. If anchor overflows R, drop its current
            // farthest out-neighbour to make room (keep the new connectivity
            // edge, since it is essential for reachability).
            if !self.graph[anchor as usize].contains(&target) {
                self.graph[anchor as usize].push(target);
                if self.graph[anchor as usize].len() > self.cfg.degree_r {
                    // Remove the farthest existing neighbour that is *not* the
                    // newly added target.
                    let a = anchor;
                    let mut farthest_idx: Option<usize> = None;
                    let mut farthest_d = f32::NEG_INFINITY;
                    for (idx, &v) in self.graph[a as usize].iter().enumerate() {
                        if v == target {
                            continue;
                        }
                        let d = self.dist_nodes(a, v);
                        if d > farthest_d {
                            farthest_d = d;
                            farthest_idx = Some(idx);
                        }
                    }
                    if let Some(idx) = farthest_idx {
                        self.graph[a as usize].remove(idx);
                    }
                }
            }
        }
        Ok(())
    }

    /// Approximate top-`k` search using a greedy beam search (search-list size
    /// `max(L, k)`) seeded from the navigating node. Returns `(id, dist²)`
    /// ascending.
    ///
    /// # Errors
    /// - [`AnnError::InvalidK`] if `k == 0`.
    /// - [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`AnnError::IndexEmpty`] if the graph is empty.
    pub fn search(&self, query: &[f32], k: usize) -> AnnResult<Vec<(u32, f32)>> {
        self.search_with_l(query, k, self.cfg.search_l)
    }

    /// Like [`Self::search`] but with an explicit search-list size `l` (clamped
    /// to `>= k`). Larger `l` explores more of the graph, raising recall.
    ///
    /// # Errors
    /// Same as [`Self::search`]; additionally treats `l == 0` as `k`.
    pub fn search_with_l(&self, query: &[f32], k: usize, l: usize) -> AnnResult<Vec<(u32, f32)>> {
        if k == 0 {
            return Err(AnnError::InvalidK {
                k,
                n: self.graph.len(),
            });
        }
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if self.graph.is_empty() {
            return Err(AnnError::IndexEmpty);
        }
        let eff_l = l.max(k);
        let mut res = self.greedy_visit_graph(query, self.navigating_node, eff_l, None);
        res.truncate(k.min(res.len()));
        Ok(res)
    }

    /// Run a greedy search and record the sequence of accepted *best-so-far*
    /// distances to the query (used to verify monotone descent of the path).
    /// Returns the descent trace (one entry per strict improvement, ascending in
    /// time, descending in value).
    ///
    /// # Errors
    /// - [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`AnnError::IndexEmpty`] if the graph is empty.
    pub fn search_descent_trace(&self, query: &[f32], l: usize) -> AnnResult<Vec<f32>> {
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if self.graph.is_empty() {
            return Err(AnnError::IndexEmpty);
        }
        let eff_l = l.max(1);
        let start = self.navigating_node;
        let mut frontier: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
        let mut visited: HashSet<u32> = HashSet::new();
        let mut consumed: HashSet<u32> = HashSet::new();
        let mut best_dist = self.dist_query(query, start);
        let mut trace = vec![best_dist];
        frontier.push(Reverse((OrdF32(best_dist), start)));
        visited.insert(start);
        let mut expanded = 0usize;

        while let Some(Reverse((OrdF32(_), c_id))) = frontier.pop() {
            if !consumed.insert(c_id) {
                continue;
            }
            expanded += 1;
            if expanded > eff_l {
                break;
            }
            for &nbr in &self.graph[c_id as usize] {
                if !visited.insert(nbr) {
                    continue;
                }
                let d = self.dist_query(query, nbr);
                frontier.push(Reverse((OrdF32(d), nbr)));
                if d < best_dist {
                    best_dist = d;
                    trace.push(best_dist);
                }
            }
        }
        Ok(trace)
    }

    /// BFS reachability from the navigating node; returns the number of reached
    /// nodes. Used by connectivity tests.
    #[must_use]
    pub fn reachable_count(&self) -> usize {
        let n = self.graph.len();
        if n == 0 {
            return 0;
        }
        let mut reached = vec![false; n];
        let mut q: VecDeque<u32> = VecDeque::new();
        let nav = self.navigating_node;
        reached[nav as usize] = true;
        q.push_back(nav);
        while let Some(u) = q.pop_front() {
            for &v in &self.graph[u as usize] {
                if !reached[v as usize] {
                    reached[v as usize] = true;
                    q.push_back(v);
                }
            }
        }
        reached.iter().filter(|&&b| b).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::l2::l2_sq;

    fn clustered_data(rng: &mut LcgRng) -> (Vec<f32>, usize, usize) {
        // 4 well-separated 4-D clusters, 25 points each => n = 100.
        let dim = 4;
        let centres = [
            [0.0_f32, 0.0, 0.0, 0.0],
            [50.0, 0.0, 0.0, 0.0],
            [0.0, 50.0, 0.0, 0.0],
            [25.0, 25.0, 50.0, 0.0],
        ];
        let mut data = Vec::new();
        for c in centres.iter() {
            for _ in 0..25 {
                for &cx in c.iter().take(dim) {
                    data.push(cx + (rng.next_f32() - 0.5) * 4.0);
                }
            }
        }
        (data, 100, dim)
    }

    fn brute_force_topk(data: &[f32], n: usize, dim: usize, query: &[f32], k: usize) -> Vec<usize> {
        let mut d: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = &data[i * dim..(i + 1) * dim];
                (i, l2_sq(query, v).expect("l2_sq should succeed"))
            })
            .collect();
        d.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        d.truncate(k);
        d.into_iter().map(|(i, _)| i).collect()
    }

    #[test]
    fn build_sets_metadata() {
        let mut rng = LcgRng::new(1);
        let (data, n, dim) = clustered_data(&mut rng);
        let idx = NsgIndex::build(&data, n, dim, NsgConfig::default(), &mut rng)
            .expect("value should be present");
        assert_eq!(idx.len(), n);
        assert_eq!(idx.dim(), dim);
        assert!(!idx.is_empty());
        assert!((idx.navigating_node() as usize) < n);
    }

    // LOAD-BEARING (a): graph is CONNECTED from the navigating node (BFS reaches all).
    #[test]
    fn graph_connected_from_navigating_node() {
        let mut rng = LcgRng::new(7);
        let (data, n, dim) = clustered_data(&mut rng);
        let idx = NsgIndex::build(&data, n, dim, NsgConfig::default(), &mut rng)
            .expect("value should be present");
        assert_eq!(
            idx.reachable_count(),
            n,
            "navigating node must reach all {n} nodes; reached {}",
            idx.reachable_count()
        );
    }

    // LOAD-BEARING (b): recall@k vs brute-force is high (>0.9) on clustered data.
    #[test]
    fn recall_high_on_clustered_data() {
        let mut rng = LcgRng::new(11);
        let (data, n, dim) = clustered_data(&mut rng);
        let cfg = NsgConfig {
            degree_r: 24,
            search_l: 64,
            knn_k: 24,
            nndescent_iters: 20,
        };
        let idx = NsgIndex::build(&data, n, dim, cfg, &mut rng).expect("build should succeed");

        let k = 10;
        let n_queries = 20;
        let mut hits = 0usize;
        let mut q_rng = LcgRng::new(999);
        for _ in 0..n_queries {
            // Query = a real point + small jitter so ground truth is meaningful.
            let base = (q_rng.next_u32() as usize) % n;
            let mut query: Vec<f32> = data[base * dim..(base + 1) * dim].to_vec();
            for v in query.iter_mut() {
                *v += (q_rng.next_f32() - 0.5) * 0.5;
            }
            let gt: HashSet<usize> = brute_force_topk(&data, n, dim, &query, k)
                .into_iter()
                .collect();
            let got = idx
                .search_with_l(&query, k, 64)
                .expect("search_with_l should succeed");
            hits += got
                .iter()
                .filter(|&&(id, _)| gt.contains(&(id as usize)))
                .count();
        }
        let recall = hits as f32 / (n_queries * k) as f32;
        assert!(recall > 0.9, "recall@{k} = {recall:.3} <= 0.9");
    }

    // LOAD-BEARING (c): MRNG occlusion pruning works — an edge to an occluded
    // point is dropped. Construct a 3-point case where b is occluded by a.
    #[test]
    fn mrng_occlusion_drops_occluded_edge() {
        // p at origin, a close on the +x axis, b farther on the +x axis.
        // Candidate order from p: a (d=1), b (d=4).
        // Accept a. Then test b: dist(a,b)=1 < dist(p,b)=4 => b is OCCLUDED by a
        // and must be dropped. So p keeps only a.
        let dim = 2;
        let data = vec![
            0.0_f32, 0.0, // p = 0
            1.0, 0.0, // a = 1
            2.0, 0.0, // b = 2
        ];
        // Build a tiny index just to access the geometry helpers.
        let mut rng = LcgRng::new(3);
        let cfg = NsgConfig {
            degree_r: 8,
            search_l: 8,
            knn_k: 2,
            nndescent_iters: 4,
        };
        let idx = NsgIndex::build(&data, 3, dim, cfg, &mut rng).expect("build should succeed");
        let p = 0u32;
        // Pool sorted ascending by dist(p,·): a then b.
        let pool = vec![(1u32, 1.0_f32), (2u32, 4.0_f32)];
        let chosen = idx.mrng_prune(p, &pool, 8);
        assert!(chosen.contains(&1), "a (id=1) must be accepted: {chosen:?}");
        assert!(
            !chosen.contains(&2),
            "b (id=2) is occluded by a and must be dropped: {chosen:?}"
        );
        assert_eq!(chosen, vec![1u32]);
    }

    // LOAD-BEARING (c'): the non-occluded case is kept. A point off-axis is NOT
    // occluded and must be retained.
    #[test]
    fn mrng_keeps_non_occluded_edge() {
        // p at origin, a on +x, c on +y. dist(a,c) = 2 > dist(p,c) = 1, so c is
        // NOT occluded by a and must be accepted.
        let dim = 2;
        let data = vec![
            0.0_f32, 0.0, // p = 0
            1.0, 0.0, // a = 1
            0.0, 1.0, // c = 2
        ];
        let mut rng = LcgRng::new(5);
        let cfg = NsgConfig {
            degree_r: 8,
            search_l: 8,
            knn_k: 2,
            nndescent_iters: 4,
        };
        let idx = NsgIndex::build(&data, 3, dim, cfg, &mut rng).expect("build should succeed");
        let pool = vec![(1u32, 1.0_f32), (2u32, 1.0_f32)];
        let chosen = idx.mrng_prune(0, &pool, 8);
        assert!(chosen.contains(&1) && chosen.contains(&2), "{chosen:?}");
    }

    // LOAD-BEARING (d): out-degree <= R for every node.
    #[test]
    fn out_degree_bounded_by_r() {
        let mut rng = LcgRng::new(13);
        let (data, n, dim) = clustered_data(&mut rng);
        let r = 16;
        let cfg = NsgConfig {
            degree_r: r,
            search_l: 40,
            knn_k: 16,
            nndescent_iters: 12,
        };
        let idx = NsgIndex::build(&data, n, dim, cfg, &mut rng).expect("build should succeed");
        for id in 0..n as u32 {
            assert!(
                idx.neighbors(id).len() <= r,
                "node {id} has out-degree {} > R={r}",
                idx.neighbors(id).len()
            );
        }
    }

    // LOAD-BEARING (e): greedy search distance-to-query is monotone
    // non-increasing along the path on clean (well-separated) data.
    #[test]
    fn greedy_descent_is_monotone() {
        let mut rng = LcgRng::new(17);
        let (data, n, dim) = clustered_data(&mut rng);
        let cfg = NsgConfig {
            degree_r: 24,
            search_l: 64,
            knn_k: 24,
            nndescent_iters: 20,
        };
        let idx = NsgIndex::build(&data, n, dim, cfg, &mut rng).expect("build should succeed");
        // Query a real point; the best-so-far trace must be strictly descending.
        let query = &data[42 * dim..43 * dim];
        let trace = idx
            .search_descent_trace(query, 64)
            .expect("search_descent_trace should succeed");
        assert!(!trace.is_empty());
        for w in trace.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-6,
                "best-so-far must be non-increasing: {trace:?}"
            );
        }
        // And it should reach (near) zero since the query is an exact point.
        assert!(
            *trace.last().expect("last should succeed") < 1e-3,
            "descent should reach the exact point: last={}",
            trace.last().expect("last should succeed")
        );
    }

    // LOAD-BEARING (f): larger search-L => recall increases (monotone, in the
    // weak sense recall(L_large) >= recall(L_small)).
    #[test]
    fn larger_l_does_not_reduce_recall() {
        let mut rng = LcgRng::new(19);
        let (data, n, dim) = clustered_data(&mut rng);
        // Use a sparse k-NN seed so small L genuinely under-explores.
        let cfg = NsgConfig {
            degree_r: 12,
            search_l: 24,
            knn_k: 10,
            nndescent_iters: 12,
        };
        let idx = NsgIndex::build(&data, n, dim, cfg, &mut rng).expect("build should succeed");

        let k = 10;
        let n_queries = 25;
        let measure_recall = |l: usize| -> f32 {
            let mut q_rng = LcgRng::new(2024);
            let mut hits = 0usize;
            for _ in 0..n_queries {
                let base = (q_rng.next_u32() as usize) % n;
                let mut query: Vec<f32> = data[base * dim..(base + 1) * dim].to_vec();
                for v in query.iter_mut() {
                    *v += (q_rng.next_f32() - 0.5) * 2.0;
                }
                let gt: HashSet<usize> = brute_force_topk(&data, n, dim, &query, k)
                    .into_iter()
                    .collect();
                let got = idx
                    .search_with_l(&query, k, l)
                    .expect("search_with_l should succeed");
                hits += got
                    .iter()
                    .filter(|&&(id, _)| gt.contains(&(id as usize)))
                    .count();
            }
            hits as f32 / (n_queries * k) as f32
        };
        let recall_small = measure_recall(12);
        let recall_large = measure_recall(80);
        assert!(
            recall_large >= recall_small - 1e-6,
            "recall(L=80)={recall_large:.3} < recall(L=12)={recall_small:.3}"
        );
    }

    #[test]
    fn single_point_index() {
        let mut rng = LcgRng::new(23);
        let data = vec![3.0_f32, 4.0];
        let idx = NsgIndex::build(&data, 1, 2, NsgConfig::default(), &mut rng)
            .expect("value should be present");
        assert_eq!(idx.len(), 1);
        let res = idx
            .search(&[3.0_f32, 4.0], 1)
            .expect("search should succeed");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 0);
        assert!(res[0].1.abs() < 1e-6);
    }

    #[test]
    fn search_finds_exact_self() {
        let mut rng = LcgRng::new(29);
        let (data, n, dim) = clustered_data(&mut rng);
        let idx = NsgIndex::build(&data, n, dim, NsgConfig::default(), &mut rng)
            .expect("value should be present");
        for &probe in &[0usize, 30, 60, 99] {
            let q = &data[probe * dim..(probe + 1) * dim];
            let res = idx
                .search_with_l(q, 1, 64)
                .expect("search_with_l should succeed");
            assert!(!res.is_empty());
            let d = l2_sq(q, idx.point(res[0].0)).expect("value should be present");
            assert!(d < 1e-4, "probe={probe} found_id={} d={d}", res[0].0);
        }
    }

    #[test]
    fn deterministic_build_same_seed() {
        let mut rng_a = LcgRng::new(31);
        let mut rng_b = LcgRng::new(31);
        let (data, n, dim) = {
            let mut r = LcgRng::new(31);
            clustered_data(&mut r)
        };
        let a = NsgIndex::build(&data, n, dim, NsgConfig::default(), &mut rng_a)
            .expect("value should be present");
        let b = NsgIndex::build(&data, n, dim, NsgConfig::default(), &mut rng_b)
            .expect("value should be present");
        assert_eq!(a.navigating_node(), b.navigating_node());
        for id in 0..n as u32 {
            assert_eq!(a.neighbors(id), b.neighbors(id), "id={id}");
        }
    }

    #[test]
    fn err_empty_input() {
        let mut rng = LcgRng::new(37);
        assert!(matches!(
            NsgIndex::build(&[], 0, 2, NsgConfig::default(), &mut rng),
            Err(AnnError::EmptyInput)
        ));
    }

    #[test]
    fn err_dim_zero() {
        let mut rng = LcgRng::new(41);
        assert!(matches!(
            NsgIndex::build(&[], 1, 0, NsgConfig::default(), &mut rng),
            Err(AnnError::InvalidVectorDim { dim: 0 })
        ));
    }

    #[test]
    fn err_data_length_mismatch() {
        let mut rng = LcgRng::new(43);
        assert!(matches!(
            NsgIndex::build(&[0.0_f32, 1.0, 2.0], 2, 2, NsgConfig::default(), &mut rng),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_zero_config() {
        let mut rng = LcgRng::new(47);
        let (data, n, dim) = clustered_data(&mut rng);
        let cfg = NsgConfig {
            degree_r: 0,
            search_l: 10,
            knn_k: 10,
            nndescent_iters: 5,
        };
        assert!(matches!(
            NsgIndex::build(&data, n, dim, cfg, &mut rng),
            Err(AnnError::Internal { .. })
        ));
    }

    #[test]
    fn err_search_k_zero() {
        let mut rng = LcgRng::new(53);
        let (data, n, dim) = clustered_data(&mut rng);
        let idx = NsgIndex::build(&data, n, dim, NsgConfig::default(), &mut rng)
            .expect("value should be present");
        assert!(matches!(
            idx.search(&data[0..dim], 0),
            Err(AnnError::InvalidK { k: 0, .. })
        ));
    }

    #[test]
    fn err_search_wrong_dim() {
        let mut rng = LcgRng::new(59);
        let (data, n, dim) = clustered_data(&mut rng);
        let idx = NsgIndex::build(&data, n, dim, NsgConfig::default(), &mut rng)
            .expect("value should be present");
        let bad = vec![0.0_f32; dim + 1];
        assert!(matches!(
            idx.search(&bad, 5),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn search_k_greater_than_n() {
        let mut rng = LcgRng::new(61);
        let data = vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0];
        let idx = NsgIndex::build(&data, 3, 2, NsgConfig::default(), &mut rng)
            .expect("value should be present");
        let res = idx
            .search_with_l(&[0.0_f32, 0.0], 100, 10)
            .expect("search_with_l should succeed");
        assert_eq!(res.len(), 3);
    }
}
