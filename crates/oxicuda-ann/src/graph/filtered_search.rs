//! Filtered ANN search — Filtered-DiskANN style label-constrained search.
//!
//! Reference: Siddharth Gollapudi, Neel Karia, Varun Sivashankar, Ravishankar
//! Krishnaswamy, Nikit Begwani, Swapnil Raz, Yiyong Lin, Yin Zhang, Neelam
//! Mahapatro, Premkumar Srinivasan, Amit Singh, Harsha Vardhan Simhadri,
//! *"Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search
//! with Filters"*, WWW 2023.
//!
//! Each indexed point carries a **label set** (a set of `u32` label ids). A
//! query supplies a [`FilterPredicate`] — a required label, an AND of labels, an
//! OR of labels, or "match everything". The index returns the nearest neighbours
//! that **satisfy the predicate**, using *filter-aware traversal*:
//!
//! * **Per-label entry points.** For every label we record a small set of entry
//!   nodes carrying that label (the medoid of the label's sub-population plus a
//!   few extras). Filtered search starts from the entry points of the labels
//!   named by the predicate, so it lands directly in the relevant region.
//! * **Filter-aware collection.** During the greedy graph walk we *traverse*
//!   every reachable node (to stay navigable) but only *collect* nodes whose
//!   label set satisfies the predicate. The result therefore contains only
//!   matching points.
//!
//! The underlying proximity graph is a single unfiltered Vamana-style graph
//! (α-pruned RobustPrune + greedy build), which keeps the whole index navigable;
//! the *filtering* happens entirely at query time plus the per-label entry-point
//! seeding. This matches the "FilteredVamana with a stitched/seeded search"
//! design from the paper while staying within a compact pure-Rust core.
//!
//! All ordering uses **squared** Euclidean distance (L2²); results are
//! `(u32, f32)` = `(id, dist²)` ascending.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::knn_graph::knn_graph::KnnGraph;

/// Totally-ordered `f32` wrapper for binary heaps (NaN sinks to "largest").
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

/// A label-set membership predicate evaluated against a point's label set.
#[derive(Debug, Clone)]
pub enum FilterPredicate {
    /// Match every point (degenerates to ordinary unfiltered search).
    Any,
    /// The point's label set must contain this single label.
    Has(u32),
    /// The point's label set must contain **all** of these labels (AND).
    All(Vec<u32>),
    /// The point's label set must contain **at least one** of these labels (OR).
    AnyOf(Vec<u32>),
}

impl FilterPredicate {
    /// Evaluate the predicate against a point's (sorted, deduplicated) label set.
    #[must_use]
    pub fn matches(&self, labels: &[u32]) -> bool {
        match self {
            FilterPredicate::Any => true,
            FilterPredicate::Has(l) => labels.binary_search(l).is_ok(),
            FilterPredicate::All(req) => req.iter().all(|l| labels.binary_search(l).is_ok()),
            FilterPredicate::AnyOf(req) => req.iter().any(|l| labels.binary_search(l).is_ok()),
        }
    }

    /// The labels that should seed the search (used to pick entry points).
    /// `Any` returns an empty set → the global entry point is used.
    fn seed_labels(&self) -> Vec<u32> {
        match self {
            FilterPredicate::Any => Vec::new(),
            FilterPredicate::Has(l) => vec![*l],
            FilterPredicate::All(req) | FilterPredicate::AnyOf(req) => req.clone(),
        }
    }
}

/// Build configuration for a [`FilteredIndex`].
#[derive(Debug, Clone)]
pub struct FilteredConfig {
    /// Maximum out-degree `R` of the proximity graph. Must be `>= 1`.
    pub degree_r: usize,
    /// Build/search list size `L`. Must be `>= 1`.
    pub search_l: usize,
    /// Number of neighbours `k` in the seed approximate k-NN graph. Must be
    /// `>= 1`.
    pub knn_k: usize,
}

impl Default for FilteredConfig {
    fn default() -> Self {
        Self {
            degree_r: 24,
            search_l: 64,
            knn_k: 20,
        }
    }
}

/// Filtered ANN index: a navigable graph + per-point labels + per-label entry
/// points.
pub struct FilteredIndex {
    /// Flat `n × dim` row-major point storage.
    points: Vec<f32>,
    /// `graph[p]` is the bounded out-neighbour list of node `p`.
    graph: Vec<Vec<u32>>,
    /// `labels[p]` is node `p`'s sorted, deduplicated label set.
    labels: Vec<Vec<u32>>,
    /// Per-label entry points (a few medoid-ish nodes carrying that label).
    label_entry: HashMap<u32, Vec<u32>>,
    /// Global entry point (medoid over all points) for `Any` queries.
    global_entry: u32,
    /// Vector dimensionality.
    dim: usize,
    /// Maximum out-degree `R`.
    degree_r: usize,
    /// Build/search list size `L`.
    search_l: usize,
    /// Number of entry points stored per label.
    entries_per_label: usize,
}

impl FilteredIndex {
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

    /// Read-only access to a node's label set.
    #[must_use]
    pub fn labels(&self, id: u32) -> &[u32] {
        let i = id as usize;
        if i < self.labels.len() {
            &self.labels[i]
        } else {
            &[]
        }
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

    /// Entry points recorded for `label`, or an empty slice when unknown.
    #[must_use]
    pub fn entry_points(&self, label: u32) -> &[u32] {
        self.label_entry
            .get(&label)
            .map_or(&[][..], |v| v.as_slice())
    }

    fn point(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.points[s..s + self.dim]
    }

    fn dist_query(&self, query: &[f32], id: u32) -> f32 {
        let v = self.point(id);
        query
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    fn dist_nodes(&self, a: u32, b: u32) -> f32 {
        let va = self.point(a);
        let vb = self.point(b);
        va.iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    /// Medoid over a subset of point ids: the member closest to the subset's
    /// centroid. Returns `None` for an empty subset.
    fn subset_medoid(&self, ids: &[u32]) -> Option<u32> {
        if ids.is_empty() {
            return None;
        }
        let mut centroid = vec![0.0_f32; self.dim];
        for &id in ids {
            for (c, &x) in centroid.iter_mut().zip(self.point(id).iter()) {
                *c += x;
            }
        }
        let inv = 1.0 / ids.len() as f32;
        for c in centroid.iter_mut() {
            *c *= inv;
        }
        let mut best = ids[0];
        let mut best_d = f32::INFINITY;
        for &id in ids {
            let d = self.dist_query(&centroid, id);
            if d < best_d {
                best_d = d;
                best = id;
            }
        }
        Some(best)
    }

    /// Greedy best-first search over `self.graph` seeded from `start`, returning
    /// the visited set as `(id, dist²)` ascending by distance to `query`,
    /// bounded to `l`. `exclude` (when set) is never collected nor expanded
    /// (used during the build to keep a node out of its own candidate pool).
    fn greedy_visit(
        &self,
        query: &[f32],
        start: u32,
        l: usize,
        exclude: Option<u32>,
    ) -> Vec<(u32, f32)> {
        let n = self.graph.len();
        if n == 0 || l == 0 {
            return Vec::new();
        }
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
        let mut frontier: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
        let mut result: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
        let mut visited: HashSet<u32> = HashSet::new();
        let mut consumed: HashSet<u32> = HashSet::new();

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
            for &nbr in &self.graph[c_id as usize] {
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

    /// RobustPrune (DiskANN Algorithm 2) bounding out-degree to `R` using slack
    /// `alpha`. `candidates` are `(id, dist_from_p)` pairs.
    fn robust_prune(&self, p: u32, candidates: &mut Vec<(u32, f32)>, alpha: f32) -> Vec<u32> {
        candidates.retain(|&(v, _)| v != p);
        if candidates.is_empty() {
            return Vec::new();
        }
        candidates.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        let r = self.degree_r;
        let mut result: Vec<u32> = Vec::with_capacity(r);
        let mut alive = vec![true; candidates.len()];
        let mut i = 0;
        while i < candidates.len() && result.len() < r {
            if !alive[i] {
                i += 1;
                continue;
            }
            let (v_star, _) = candidates[i];
            result.push(v_star);
            alive[i] = false;
            for j in (i + 1)..candidates.len() {
                if !alive[j] {
                    continue;
                }
                let (v_prime, d_p_vprime) = candidates[j];
                let d_vstar_vprime = self.dist_nodes(v_star, v_prime);
                if alpha * d_vstar_vprime <= d_p_vprime {
                    alive[j] = false;
                }
            }
            i += 1;
        }
        result
    }

    /// Build a filtered index from `data` (row-major `n × dim`) and a
    /// per-point label set `point_labels` (`point_labels[p]` = labels of node
    /// `p`; will be sorted/deduplicated internally).
    ///
    /// # Errors
    /// - [`AnnError::EmptyInput`] if `n == 0`.
    /// - [`AnnError::InvalidVectorDim`] if `dim == 0`.
    /// - [`AnnError::DimensionMismatch`] if `data.len() != n * dim`.
    /// - [`AnnError::Internal`] if `point_labels.len() != n`, or if any of
    ///   `cfg.degree_r`, `cfg.search_l`, `cfg.knn_k` is `0`.
    pub fn build(
        data: &[f32],
        n: usize,
        dim: usize,
        point_labels: &[Vec<u32>],
        cfg: FilteredConfig,
        rng: &mut LcgRng,
    ) -> AnnResult<Self> {
        let FilteredConfig {
            degree_r,
            search_l,
            knn_k,
        } = cfg;
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
        if point_labels.len() != n {
            return Err(AnnError::Internal {
                msg: format!(
                    "point_labels length {} must equal n={n}",
                    point_labels.len()
                ),
            });
        }
        if degree_r == 0 || search_l == 0 || knn_k == 0 {
            return Err(AnnError::Internal {
                msg: "degree_r, search_l and knn_k must all be >= 1".to_string(),
            });
        }

        // Normalise label sets (sorted + deduplicated for binary-search match).
        let labels: Vec<Vec<u32>> = point_labels
            .iter()
            .map(|ls| {
                let mut v = ls.clone();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();

        let mut index = Self {
            points: data.to_vec(),
            graph: vec![Vec::new(); n],
            labels,
            label_entry: HashMap::new(),
            global_entry: 0,
            dim,
            degree_r,
            search_l,
            entries_per_label: 4,
        };

        // Global entry = medoid over all points (computed first so the
        // navigable build can be seeded from it).
        let all: Vec<u32> = (0..n as u32).collect();
        index.global_entry = index.subset_medoid(&all).unwrap_or(0);

        // Build a navigable proximity graph (FilteredVamana core). We seed every
        // node's initial out-edges from an approximate k-NN graph (RobustPrune),
        // then make the graph navigable from the global entry with a Vamana-style
        // refinement pass: greedy-search from the entry towards each node and
        // RobustPrune the visited set into its out-neighbours, adding back-edges.
        // The greedy-seeded refinement is what guarantees a single-entry walk can
        // reach the whole graph (high unfiltered recall).
        if n > 1 {
            let alpha = 1.2_f32;
            let knn = KnnGraph::build_nn_descent(data, n, dim, knn_k, 12, 0.001, rng);
            for p in 0..n as u32 {
                let mut cand: Vec<(u32, f32)> = knn
                    .neighbors(p as usize)
                    .iter()
                    .map(|&(id, d)| (id, d))
                    .collect();
                index.graph[p as usize] = index.robust_prune(p, &mut cand, alpha);
            }
            // Symmetrise so the seed graph is undirected-ish before refinement.
            for p in 0..n as u32 {
                let nbrs = index.graph[p as usize].clone();
                for q in nbrs {
                    let qi = q as usize;
                    if q != p && !index.graph[qi].contains(&p) {
                        index.graph[qi].push(p);
                        if index.graph[qi].len() > degree_r {
                            let mut c: Vec<(u32, f32)> = index.graph[qi]
                                .iter()
                                .filter(|&&v| v != q)
                                .map(|&v| (v, index.dist_nodes(q, v)))
                                .collect();
                            index.graph[qi] = index.robust_prune(q, &mut c, alpha);
                        }
                    }
                }
            }
            // Navigability refinement passes from the global entry.
            let entry = index.global_entry;
            let build_l = search_l.max(degree_r);
            for _pass in 0..2 {
                for p in 0..n as u32 {
                    let q_vec: Vec<f32> = index.point(p).to_vec();
                    let mut visited = index.greedy_visit(&q_vec, entry, build_l, Some(p));
                    // Union current out-neighbours so we never lose good edges.
                    let mut seen: HashSet<u32> = visited.iter().map(|&(id, _)| id).collect();
                    for &nb in &index.graph[p as usize] {
                        if nb != p && seen.insert(nb) {
                            visited.push((nb, index.dist_nodes(p, nb)));
                        }
                    }
                    index.graph[p as usize] = index.robust_prune(p, &mut visited, alpha);
                    // Back-edges with overflow re-pruning.
                    let chosen = index.graph[p as usize].clone();
                    for q in chosen {
                        let qi = q as usize;
                        if q != p && !index.graph[qi].contains(&p) {
                            index.graph[qi].push(p);
                            if index.graph[qi].len() > degree_r {
                                let mut c: Vec<(u32, f32)> = index.graph[qi]
                                    .iter()
                                    .filter(|&&v| v != q)
                                    .map(|&v| (v, index.dist_nodes(q, v)))
                                    .collect();
                                index.graph[qi] = index.robust_prune(q, &mut c, alpha);
                            }
                        }
                    }
                }
            }
        }

        // Per-label entry points: for each label, gather its members, pick the
        // medoid plus a few additional members as extra seeds.
        let mut by_label: HashMap<u32, Vec<u32>> = HashMap::new();
        for (p, ls) in index.labels.iter().enumerate() {
            for &lab in ls {
                by_label.entry(lab).or_default().push(p as u32);
            }
        }
        for (lab, members) in by_label {
            let mut entries: Vec<u32> = Vec::new();
            if let Some(med) = index.subset_medoid(&members) {
                entries.push(med);
            }
            // Add a few deterministic extra members (stride sampling) so search
            // has multiple seeds in the matching sub-population.
            if members.len() > 1 {
                let stride = (members.len() / index.entries_per_label).max(1);
                let mut i = 0;
                while i < members.len() && entries.len() < index.entries_per_label {
                    let cand = members[i];
                    if !entries.contains(&cand) {
                        entries.push(cand);
                    }
                    i += stride;
                }
            }
            index.label_entry.insert(lab, entries);
        }

        Ok(index)
    }

    /// Collect the seed nodes for a query under `predicate`: the union of entry
    /// points of every seed label, falling back to the global entry point.
    fn seeds_for(&self, predicate: &FilterPredicate) -> Vec<u32> {
        let mut seeds: Vec<u32> = Vec::new();
        for lab in predicate.seed_labels() {
            if let Some(eps) = self.label_entry.get(&lab) {
                for &e in eps {
                    if !seeds.contains(&e) {
                        seeds.push(e);
                    }
                }
            }
        }
        if seeds.is_empty() {
            seeds.push(self.global_entry);
        }
        seeds
    }

    /// Filtered greedy beam search: traverse the whole reachable graph but only
    /// collect nodes satisfying `predicate`. Returns up to `k` matching nodes
    /// ascending by squared-L2 distance.
    ///
    /// # Errors
    /// - [`AnnError::InvalidK`] if `k == 0`.
    /// - [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`AnnError::IndexEmpty`] if the graph is empty.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        predicate: &FilterPredicate,
    ) -> AnnResult<Vec<(u32, f32)>> {
        self.search_with_l(query, k, predicate, self.search_l)
    }

    /// Like [`Self::search`] with an explicit search-list size `l` (clamped to
    /// `>= k`). The `l` bounds the *result* beam; traversal always follows graph
    /// edges so the walk stays navigable even when matches are sparse.
    ///
    /// # Errors
    /// Same as [`Self::search`].
    pub fn search_with_l(
        &self,
        query: &[f32],
        k: usize,
        predicate: &FilterPredicate,
        l: usize,
    ) -> AnnResult<Vec<(u32, f32)>> {
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
        let seeds = self.seeds_for(predicate);

        // frontier: min-heap by distance (drives the traversal over ALL nodes).
        let mut frontier: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
        // result: max-heap of MATCHING nodes only, bounded to eff_l.
        let mut result: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
        let mut visited: HashSet<u32> = HashSet::new();
        let mut consumed: HashSet<u32> = HashSet::new();

        for &s in &seeds {
            if visited.insert(s) {
                let d = self.dist_query(query, s);
                frontier.push(Reverse((OrdF32(d), s)));
                if predicate.matches(&self.labels[s as usize]) {
                    self.push_result(&mut result, d, s, eff_l);
                }
            }
        }

        while let Some(Reverse((OrdF32(c_dist), c_id))) = frontier.pop() {
            if !consumed.insert(c_id) {
                continue;
            }
            // Early stop: if the frontier head is worse than our worst *matching*
            // result and the beam is already full, no closer match can appear
            // along this branch's head. (Conservative: we still expand if the
            // result is not yet full, to keep finding matches.)
            if result.len() >= eff_l {
                let worst = result.peek().map_or(f32::INFINITY, |(OrdF32(d), _)| *d);
                if c_dist > worst {
                    break;
                }
            }
            for &nbr in &self.graph[c_id as usize] {
                if !visited.insert(nbr) {
                    continue;
                }
                let d = self.dist_query(query, nbr);
                // Always push to frontier so traversal continues through
                // non-matching nodes (filter-aware traversal).
                frontier.push(Reverse((OrdF32(d), nbr)));
                if predicate.matches(&self.labels[nbr as usize]) {
                    self.push_result(&mut result, d, nbr, eff_l);
                }
            }
        }

        let mut out: Vec<(u32, f32)> = result.into_iter().map(|(OrdF32(d), id)| (id, d)).collect();
        out.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        out.truncate(k.min(out.len()));
        Ok(out)
    }

    /// Ordinary (unfiltered) approximate top-`k` search: a plain greedy beam
    /// search from the global entry point that collects every visited node.
    /// Provided so callers can express "no filter" explicitly; it is equivalent
    /// to [`Self::search`] with [`FilterPredicate::Any`].
    ///
    /// # Errors
    /// - [`AnnError::InvalidK`] if `k == 0`.
    /// - [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`AnnError::IndexEmpty`] if the graph is empty.
    pub fn search_unfiltered(
        &self,
        query: &[f32],
        k: usize,
        l: usize,
    ) -> AnnResult<Vec<(u32, f32)>> {
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
        let start = self.global_entry;
        let mut frontier: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
        let mut result: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
        let mut visited: HashSet<u32> = HashSet::new();
        let mut consumed: HashSet<u32> = HashSet::new();

        let d0 = self.dist_query(query, start);
        frontier.push(Reverse((OrdF32(d0), start)));
        self.push_result(&mut result, d0, start, eff_l);
        visited.insert(start);

        while let Some(Reverse((OrdF32(c_dist), c_id))) = frontier.pop() {
            if !consumed.insert(c_id) {
                continue;
            }
            if result.len() >= eff_l {
                let worst = result.peek().map_or(f32::INFINITY, |(OrdF32(d), _)| *d);
                if c_dist > worst {
                    break;
                }
            }
            for &nbr in &self.graph[c_id as usize] {
                if !visited.insert(nbr) {
                    continue;
                }
                let d = self.dist_query(query, nbr);
                frontier.push(Reverse((OrdF32(d), nbr)));
                self.push_result(&mut result, d, nbr, eff_l);
            }
        }

        let mut out: Vec<(u32, f32)> = result.into_iter().map(|(OrdF32(d), id)| (id, d)).collect();
        out.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        out.truncate(k.min(out.len()));
        Ok(out)
    }

    /// Push `(d, id)` into the bounded max-heap of matching results.
    fn push_result(&self, result: &mut BinaryHeap<(OrdF32, u32)>, d: f32, id: u32, cap: usize) {
        if result.len() < cap {
            result.push((OrdF32(d), id));
        } else {
            let worst = result.peek().map_or(f32::INFINITY, |(OrdF32(w), _)| *w);
            if d < worst {
                result.pop();
                result.push((OrdF32(d), id));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::l2::l2_sq;

    /// 6 clusters × labels. Returns (data, n, dim, labels).
    fn labelled_data(rng: &mut LcgRng) -> (Vec<f32>, usize, usize, Vec<Vec<u32>>) {
        let dim = 4;
        // 4 spatial clusters; assign labels so each label spans clusters and
        // each point gets 1–2 labels.
        let centres = [
            [0.0_f32, 0.0, 0.0, 0.0],
            [40.0, 0.0, 0.0, 0.0],
            [0.0, 40.0, 0.0, 0.0],
            [40.0, 40.0, 0.0, 0.0],
        ];
        let mut data = Vec::new();
        let mut labels = Vec::new();
        let per = 30;
        for (ci, c) in centres.iter().enumerate() {
            for j in 0..per {
                for &cx in c.iter().take(dim) {
                    data.push(cx + (rng.next_f32() - 0.5) * 3.0);
                }
                // Label scheme: label 0 = even index, label 1 = odd index,
                // label (10 + ci) = cluster tag. So every point has 2 labels.
                let mut ls = vec![if j % 2 == 0 { 0u32 } else { 1u32 }, 10 + ci as u32];
                // A rare label 99 on a couple of points only.
                if ci == 0 && j == 0 {
                    ls.push(99);
                }
                labels.push(ls);
            }
        }
        (data, centres.len() * per, dim, labels)
    }

    fn brute_filtered_topk(
        data: &[f32],
        n: usize,
        dim: usize,
        labels: &[Vec<u32>],
        query: &[f32],
        k: usize,
        pred: &FilterPredicate,
    ) -> Vec<usize> {
        let mut d: Vec<(usize, f32)> = (0..n)
            .filter(|&i| {
                let mut ls = labels[i].clone();
                ls.sort_unstable();
                ls.dedup();
                pred.matches(&ls)
            })
            .map(|i| {
                let v = &data[i * dim..(i + 1) * dim];
                (i, l2_sq(query, v).expect("l2_sq should succeed"))
            })
            .collect();
        d.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        d.truncate(k);
        d.into_iter().map(|(i, _)| i).collect()
    }

    fn build_default(rng: &mut LcgRng) -> (FilteredIndex, Vec<f32>, usize, usize, Vec<Vec<u32>>) {
        let (data, n, dim, labels) = labelled_data(rng);
        let idx = FilteredIndex::build(&data, n, dim, &labels, FilteredConfig::default(), rng)
            .expect("value should be present");
        (idx, data, n, dim, labels)
    }

    // LOAD-BEARING (a): results contain ONLY points satisfying the filter.
    #[test]
    fn results_only_satisfy_filter() {
        let mut rng = LcgRng::new(7);
        let (idx, data, _n, dim, labels) = build_default(&mut rng);
        let pred = FilterPredicate::Has(10); // cluster-0 tag
        let mut q_rng = LcgRng::new(101);
        for _ in 0..10 {
            let mut query = vec![0.0_f32; dim];
            for v in query.iter_mut() {
                *v = (q_rng.next_f32() - 0.5) * 60.0;
            }
            let res = idx.search(&query, 8, &pred).expect("search should succeed");
            for (id, _) in res {
                assert!(
                    pred.matches(&{
                        let mut l = labels[id as usize].clone();
                        l.sort_unstable();
                        l
                    }),
                    "id={id} labels={:?} must satisfy {pred:?}",
                    labels[id as usize]
                );
            }
        }
        let _ = data;
    }

    // LOAD-BEARING (b): recall@k vs brute-force-over-the-filtered-subset is high.
    #[test]
    fn recall_vs_filtered_subset_high() {
        let mut rng = LcgRng::new(11);
        let (idx, data, n, dim, labels) = build_default(&mut rng);
        let pred = FilterPredicate::Has(0); // even-index points (~half)
        let k = 10;
        let n_queries = 20;
        let mut hits = 0usize;
        let mut q_rng = LcgRng::new(202);
        for _ in 0..n_queries {
            let base = (q_rng.next_u32() as usize) % n;
            let mut query: Vec<f32> = data[base * dim..(base + 1) * dim].to_vec();
            for v in query.iter_mut() {
                *v += (q_rng.next_f32() - 0.5) * 1.0;
            }
            let gt: HashSet<usize> = brute_filtered_topk(&data, n, dim, &labels, &query, k, &pred)
                .into_iter()
                .collect();
            let got = idx
                .search_with_l(&query, k, &pred, 96)
                .expect("search_with_l should succeed");
            hits += got
                .iter()
                .filter(|&&(id, _)| gt.contains(&(id as usize)))
                .count();
        }
        let recall = hits as f32 / (n_queries * k) as f32;
        assert!(recall > 0.9, "filtered recall@{k} = {recall:.3} <= 0.9");
    }

    // LOAD-BEARING (c): a filter matching ALL points reduces to ordinary
    // unfiltered search. Two faithful checks:
    //   1. `Any` is *exactly* the unfiltered greedy walk over the same graph
    //      (the dedicated [`FilteredIndex::search_unfiltered`] convenience must
    //      agree bit-for-bit with `Any`).
    //   2. An all-matching filter (here `AnyOf([0,1])`, which every point
    //      satisfies since each point has parity label 0 or 1) recovers the true
    //      nearest neighbours, i.e. the unconstrained brute-force ground truth —
    //      the operational meaning of "reduces to ordinary search".
    #[test]
    fn any_filter_equals_unfiltered() {
        let mut rng = LcgRng::new(13);
        let (idx, data, n, dim, _labels) = build_default(&mut rng);
        let k = 10;
        let mut q_rng = LcgRng::new(303);
        let mut any_hits = 0usize;
        let mut or_hits = 0usize;
        let n_queries = 12;
        for _ in 0..n_queries {
            let base = (q_rng.next_u32() as usize) % n;
            let mut query: Vec<f32> = data[base * dim..(base + 1) * dim].to_vec();
            for v in query.iter_mut() {
                *v += (q_rng.next_f32() - 0.5) * 1.0;
            }

            // (1) `Any` must equal the dedicated unfiltered search exactly.
            let with_any = idx
                .search_with_l(&query, k, &FilterPredicate::Any, 96)
                .expect("value should be present");
            let unfiltered = idx
                .search_unfiltered(&query, k, 96)
                .expect("search_unfiltered should succeed");
            assert_eq!(
                with_any, unfiltered,
                "`Any` must equal unfiltered search: base={base}"
            );

            // (2) Both `Any` and a match-all OR recover the true neighbours
            //     (unconstrained ground truth).
            let gt: HashSet<usize> =
                brute_filtered_topk(&data, n, dim, &_labels, &query, k, &FilterPredicate::Any)
                    .into_iter()
                    .collect();
            any_hits += with_any
                .iter()
                .filter(|&&(id, _)| gt.contains(&(id as usize)))
                .count();
            let or_all = FilterPredicate::AnyOf(vec![0, 1]); // every point matches
            let with_or = idx
                .search_with_l(&query, k, &or_all, 96)
                .expect("search_with_l should succeed");
            // The OR matches all points, so its ground truth is the same.
            or_hits += with_or
                .iter()
                .filter(|&&(id, _)| gt.contains(&(id as usize)))
                .count();
        }
        let any_recall = any_hits as f32 / (n_queries * k) as f32;
        let or_recall = or_hits as f32 / (n_queries * k) as f32;
        assert!(any_recall > 0.9, "Any-filter recall {any_recall:.3} <= 0.9");
        assert!(
            or_recall > 0.9,
            "match-all OR recall {or_recall:.3} <= 0.9 (should reduce to plain search)"
        );
    }

    // LOAD-BEARING (d): a filter matching NO points => empty result, no panic.
    #[test]
    fn no_match_filter_returns_empty() {
        let mut rng = LcgRng::new(17);
        let (idx, data, _n, dim, _labels) = build_default(&mut rng);
        // Label 7777 is on no point.
        let pred = FilterPredicate::Has(7777);
        let query: Vec<f32> = data[0..dim].to_vec();
        let res = idx
            .search(&query, 10, &pred)
            .expect("search should succeed");
        assert!(res.is_empty(), "no-match filter must be empty: {res:?}");
    }

    // LOAD-BEARING (e): AND / OR multi-label filters handled correctly.
    #[test]
    fn and_or_multilabel_filters() {
        let mut rng = LcgRng::new(19);
        let (idx, data, n, dim, labels) = build_default(&mut rng);

        // AND: label 0 (even) AND label 10 (cluster 0). Both must hold.
        let and_pred = FilterPredicate::All(vec![0, 10]);
        let query: Vec<f32> = data[0..dim].to_vec();
        let res_and = idx
            .search_with_l(&query, 8, &and_pred, 96)
            .expect("search_with_l should succeed");
        assert!(!res_and.is_empty(), "AND filter should find points");
        for (id, _) in &res_and {
            let mut l = labels[*id as usize].clone();
            l.sort_unstable();
            assert!(l.binary_search(&0).is_ok() && l.binary_search(&10).is_ok());
        }
        // Cross-check the AND result count/membership against brute force.
        let gt_and: HashSet<usize> =
            brute_filtered_topk(&data, n, dim, &labels, &query, 8, &and_pred)
                .into_iter()
                .collect();
        let hits_and = res_and
            .iter()
            .filter(|&&(id, _)| gt_and.contains(&(id as usize)))
            .count();
        assert!(hits_and as f32 / gt_and.len().max(1) as f32 > 0.8);

        // OR: label 10 OR label 11 (clusters 0 or 1). Each result has one of them.
        let or_pred = FilterPredicate::AnyOf(vec![10, 11]);
        let res_or = idx
            .search_with_l(&query, 12, &or_pred, 96)
            .expect("search_with_l should succeed");
        for (id, _) in &res_or {
            let mut l = labels[*id as usize].clone();
            l.sort_unstable();
            assert!(l.binary_search(&10).is_ok() || l.binary_search(&11).is_ok());
        }
    }

    // LOAD-BEARING (f): per-filter entry points are used (and exist).
    #[test]
    fn per_filter_entry_points_used() {
        let mut rng = LcgRng::new(23);
        let (idx, _data, _n, _dim, labels) = build_default(&mut rng);
        // Each present label must have at least one entry point, and that entry
        // point must itself carry the label.
        for lab in [0u32, 1, 10, 11, 12, 13] {
            let eps = idx.entry_points(lab);
            assert!(!eps.is_empty(), "label {lab} has no entry points");
            for &e in eps {
                let mut l = labels[e as usize].clone();
                l.sort_unstable();
                assert!(
                    l.binary_search(&lab).is_ok(),
                    "entry point {e} for label {lab} does not carry it: {l:?}"
                );
            }
        }
        // The seeds chosen for a `Has(label)` predicate must be exactly that
        // label's entry points (not the global entry).
        let seeds = idx.seeds_for(&FilterPredicate::Has(12));
        let eps: HashSet<u32> = idx.entry_points(12).iter().copied().collect();
        assert!(!seeds.is_empty());
        for s in seeds {
            assert!(eps.contains(&s), "seed {s} not an entry point of label 12");
        }
    }

    // A query for the rare label 99 still finds the single matching point.
    #[test]
    fn rare_label_single_match() {
        let mut rng = LcgRng::new(29);
        let (idx, data, _n, dim, labels) = build_default(&mut rng);
        // The point carrying label 99 is the very first generated point (id 0).
        assert!(labels[0].contains(&99));
        let query: Vec<f32> = data[0..dim].to_vec();
        let res = idx
            .search(&query, 5, &FilterPredicate::Has(99))
            .expect("value should be present");
        assert_eq!(res.len(), 1, "exactly one point carries label 99");
        assert_eq!(res[0].0, 0);
        assert!(res[0].1.abs() < 1e-3);
    }

    #[test]
    fn predicate_matches_semantics() {
        let labels = vec![1u32, 3, 5, 7]; // sorted
        assert!(FilterPredicate::Any.matches(&labels));
        assert!(FilterPredicate::Has(3).matches(&labels));
        assert!(!FilterPredicate::Has(4).matches(&labels));
        assert!(FilterPredicate::All(vec![1, 5]).matches(&labels));
        assert!(!FilterPredicate::All(vec![1, 4]).matches(&labels));
        assert!(FilterPredicate::AnyOf(vec![4, 5]).matches(&labels));
        assert!(!FilterPredicate::AnyOf(vec![2, 4]).matches(&labels));
    }

    #[test]
    fn err_label_length_mismatch() {
        let mut rng = LcgRng::new(31);
        let data = vec![0.0_f32, 0.0, 1.0, 1.0];
        let labels = vec![vec![0u32]]; // only 1, but n=2
        let cfg = FilteredConfig {
            degree_r: 4,
            search_l: 8,
            knn_k: 2,
        };
        assert!(matches!(
            FilteredIndex::build(&data, 2, 2, &labels, cfg, &mut rng),
            Err(AnnError::Internal { .. })
        ));
    }

    #[test]
    fn err_empty_and_dim_zero() {
        let mut rng = LcgRng::new(37);
        let cfg = FilteredConfig {
            degree_r: 4,
            search_l: 8,
            knn_k: 2,
        };
        assert!(matches!(
            FilteredIndex::build(&[], 0, 2, &[], cfg.clone(), &mut rng),
            Err(AnnError::EmptyInput)
        ));
        assert!(matches!(
            FilteredIndex::build(&[], 1, 0, &[vec![0u32]], cfg, &mut rng),
            Err(AnnError::InvalidVectorDim { dim: 0 })
        ));
    }

    #[test]
    fn err_search_k_zero_and_dim() {
        let mut rng = LcgRng::new(41);
        let (idx, data, _n, dim, _labels) = build_default(&mut rng);
        assert!(matches!(
            idx.search(&data[0..dim], 0, &FilterPredicate::Any),
            Err(AnnError::InvalidK { k: 0, .. })
        ));
        let bad = vec![0.0_f32; dim + 2];
        assert!(matches!(
            idx.search(&bad, 5, &FilterPredicate::Any),
            Err(AnnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn single_point_with_label() {
        let mut rng = LcgRng::new(43);
        let data = vec![2.0_f32, 5.0];
        let labels = vec![vec![42u32]];
        let cfg = FilteredConfig {
            degree_r: 4,
            search_l: 8,
            knn_k: 1,
        };
        let idx = FilteredIndex::build(&data, 1, 2, &labels, cfg, &mut rng)
            .expect("build should succeed");
        let res = idx
            .search(&[2.0_f32, 5.0], 1, &FilterPredicate::Has(42))
            .expect("value should be present");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 0);
        // A non-matching filter on the only point => empty.
        let none = idx
            .search(&[2.0_f32, 5.0], 1, &FilterPredicate::Has(1))
            .expect("value should be present");
        assert!(none.is_empty());
    }
}
