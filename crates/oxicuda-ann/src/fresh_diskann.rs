//! FreshDiskANN — incremental (insert / delete) Vamana graph index.
//!
//! Reference: Aditi Singh, Suhas Jayaram Subramanya, Ravishankar Krishnaswamy,
//! Harsha Vardhan Simhadri, *"FreshDiskANN: A Fast and Accurate Graph-Based ANN
//! Index for Streaming Similarity Search"*, 2021 (arXiv:2105.09613).
//!
//! The static [`crate::vamana::VamanaIndex`] builds a navigable Vamana graph in a
//! single shot. FreshDiskANN turns that backbone into a **mutable** index that
//! supports streaming `insert` and `delete` while keeping search recall high.
//! This module implements the *in-memory* algorithmic core of the paper (the
//! "TempIndex" / RW-portion); on-SSD block layout and the LTI/StreamingMerge IO
//! pipeline are intentionally out of scope and noted in `TODO.md` as
//! hardware-gated.
//!
//! ## Operations
//!
//! * **`insert(x)`** — *Algorithm: Insert* of the paper. (1) `V ←
//!   greedy_search(x, L)` over the live graph (visited candidate set); (2)
//!   `N_out(p) ← robust_prune(p, V, α, R)` for the new node `p`; (3) for each
//!   chosen `q`, add the back-edge `q → p`, and if `|N_out(q)| > R` re-prune `q`
//!   with `robust_prune(q, N_out(q) ∪ {p}, α, R)`. New nodes reuse a free slot
//!   left behind by a consolidated deletion when one is available, otherwise
//!   they extend the arrays — this is the paper's *in-place slot reuse*.
//!
//! * **`delete(id)`** — *lazy deletion*. The id is recorded in a `DeleteList`
//!   (tombstone). Tombstoned nodes are skipped by search and never returned, but
//!   their out-edges remain in place until consolidation so that the graph stays
//!   traversable.
//!
//! * **`consolidate()`** — *Algorithm: Delete / consolidation* of the paper.
//!   For every live node `p` whose out-list points at a deleted node `d`, the
//!   edge `p → d` is replaced by edges `p → N_out(d) \ {deleted}` (bridging over
//!   the hole), then `N_out(p)` is re-pruned back to degree `R` with RobustPrune.
//!   Afterwards the deleted slots are released to the free list for reuse and the
//!   delete list is cleared.
//!
//! All distances are squared L2. Search results are `(u32, f32) = (id, dist²)`
//! ascending. The structure is deterministic: insertion order and the
//! tie-breaking rules fully determine the graph (no RNG is consulted).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::error::{AnnError, AnnResult};

/// Totally-ordered `f32` wrapper for heaps. `NaN` is treated as the largest
/// value so it never poisons ordering.
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

/// Build / search / mutation configuration for [`FreshDiskAnnIndex`].
#[derive(Debug, Clone)]
pub struct FreshDiskAnnConfig {
    /// Maximum out-degree `R` (`>= 1`).
    pub degree_r: usize,
    /// Greedy-search candidate-list size `L` used while inserting (`>= 1`).
    pub search_l: usize,
    /// RobustPrune slack `α` (`>= 1.0`, finite). Typically `1.0..=1.3`.
    pub alpha: f32,
}

impl FreshDiskAnnConfig {
    /// Validate a configuration.
    ///
    /// # Errors
    /// [`AnnError::Internal`] when `degree_r == 0`, `search_l == 0`, or `alpha`
    /// is non-finite / `< 1.0`.
    pub fn validate(&self) -> AnnResult<()> {
        if self.degree_r == 0 {
            return Err(AnnError::Internal {
                msg: "degree_r must be >= 1".to_string(),
            });
        }
        if self.search_l == 0 {
            return Err(AnnError::Internal {
                msg: "search_l must be >= 1".to_string(),
            });
        }
        if !self.alpha.is_finite() || self.alpha < 1.0 {
            return Err(AnnError::Internal {
                msg: format!("alpha must be finite and >= 1.0, got {}", self.alpha),
            });
        }
        Ok(())
    }
}

/// A mutable, incrementally-updatable Vamana graph index (FreshDiskANN core).
pub struct FreshDiskAnnIndex {
    /// Flat row-major slot storage `[capacity × dim]`. Dead slots keep stale
    /// coordinates until overwritten by a reused insert.
    points: Vec<f32>,
    /// `graph[slot]` is the bounded out-neighbour list of `slot` (empty for dead
    /// slots).
    graph: Vec<Vec<u32>>,
    /// Liveness mask: `live[slot] == false` for never-used or consolidated slots.
    live: Vec<bool>,
    /// Tombstones: ids marked deleted but not yet consolidated. Such slots are
    /// still `live == true` but are excluded from search results.
    deleted: HashSet<u32>,
    /// Free slots available for reuse by the next insert (released by
    /// [`Self::consolidate`]).
    free_slots: Vec<u32>,
    /// Deterministic search entry slot; refreshed to a live, non-deleted slot
    /// whenever the current one becomes invalid.
    entry: Option<u32>,
    /// Vector dimensionality (fixed at first insert).
    dim: usize,
    /// Cached configuration.
    cfg: FreshDiskAnnConfig,
}

impl FreshDiskAnnIndex {
    /// Create an empty index with vector dimension `dim`.
    ///
    /// # Errors
    /// [`AnnError::InvalidVectorDim`] if `dim == 0`, or any config error from
    /// [`FreshDiskAnnConfig::validate`].
    pub fn new(dim: usize, cfg: FreshDiskAnnConfig) -> AnnResult<Self> {
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim });
        }
        cfg.validate()?;
        Ok(Self {
            points: Vec::new(),
            graph: Vec::new(),
            live: Vec::new(),
            deleted: HashSet::new(),
            free_slots: Vec::new(),
            entry: None,
            dim,
            cfg,
        })
    }

    /// Number of live, non-deleted points currently searchable.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live
            .iter()
            .enumerate()
            .filter(|&(slot, &l)| l && !self.deleted.contains(&(slot as u32)))
            .count()
    }

    /// `true` when no live, non-deleted points exist.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total number of allocated slots (live + tombstoned + free).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.graph.len()
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of tombstoned (deleted-but-not-consolidated) slots.
    #[must_use]
    pub fn pending_deletions(&self) -> usize {
        self.deleted.len()
    }

    /// `true` when `id` references a live, non-deleted slot.
    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        let i = id as usize;
        i < self.live.len() && self.live[i] && !self.deleted.contains(&id)
    }

    /// Read-only out-neighbour list for `id` (empty for dead / out-of-range).
    #[must_use]
    pub fn neighbors(&self, id: u32) -> &[u32] {
        let i = id as usize;
        if i < self.graph.len() {
            &self.graph[i]
        } else {
            &[]
        }
    }

    /// Borrow the stored coordinates of slot `id`.
    fn point(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.points[s..s + self.dim]
    }

    /// Squared-L2 distance between an external `query` and slot `id`.
    fn dist_query(&self, query: &[f32], id: u32) -> f32 {
        let v = self.point(id);
        query
            .iter()
            .zip(v.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// Squared-L2 distance between two stored slots.
    fn dist_nodes(&self, a: u32, b: u32) -> f32 {
        let va = self.point(a);
        let vb = self.point(b);
        va.iter()
            .zip(vb.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    /// Pick (and remove) a slot for a new point: reuse a free slot if possible,
    /// otherwise grow the arrays.
    fn alloc_slot(&mut self, v: &[f32]) -> u32 {
        if let Some(slot) = self.free_slots.pop() {
            let s = slot as usize * self.dim;
            self.points[s..s + self.dim].copy_from_slice(v);
            self.graph[slot as usize].clear();
            self.live[slot as usize] = true;
            slot
        } else {
            let slot = self.graph.len() as u32;
            self.points.extend_from_slice(v);
            self.graph.push(Vec::new());
            self.live.push(true);
            slot
        }
    }

    /// Choose a deterministic live, non-deleted entry slot. Returns `None` when
    /// the index has no searchable points.
    fn pick_entry(&self) -> Option<u32> {
        if let Some(e) = self.entry
            && self.contains(e)
        {
            return Some(e);
        }
        (0..self.graph.len() as u32).find(|&s| self.contains(s))
    }

    /// Insert vector `x`, returning the assigned id (slot).
    ///
    /// # Errors
    /// - [`AnnError::DimensionMismatch`] if `x.len() != dim`.
    pub fn insert(&mut self, x: &[f32]) -> AnnResult<u32> {
        if x.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }

        // First live point: just allocate and become the entry.
        let entry = self.pick_entry();
        let p = self.alloc_slot(x);
        let Some(_start) = entry else {
            self.entry = Some(p);
            return Ok(p);
        };

        // 1. Greedy search the live graph with p's coordinates.
        let q_vec: Vec<f32> = self.point(p).to_vec();
        let mut visited = self.greedy_collect(&q_vec, self.cfg.search_l, Some(p));

        // 2. RobustPrune to pick p's out-neighbours.
        let pruned = self.robust_prune(p, &mut visited);
        self.graph[p as usize] = pruned.clone();

        // 3. Back-edges with re-pruning on overflow.
        for q in pruned {
            let qi = q as usize;
            if !self.graph[qi].contains(&p) {
                self.graph[qi].push(p);
            }
            if self.graph[qi].len() > self.cfg.degree_r {
                let mut cand: Vec<(u32, f32)> = self.graph[qi]
                    .iter()
                    .filter(|&&v| v != q)
                    .map(|&v| (v, self.dist_nodes(q, v)))
                    .collect();
                self.graph[qi] = self.robust_prune(q, &mut cand);
            }
        }

        if self.entry.is_none() {
            self.entry = Some(p);
        }
        Ok(p)
    }

    /// Mark `id` deleted (lazy tombstone). Idempotent; deleting a missing id is a
    /// no-op error.
    ///
    /// # Errors
    /// - [`AnnError::IdOutOfRange`] if `id` is not an allocated, live slot.
    pub fn delete(&mut self, id: u32) -> AnnResult<()> {
        let i = id as usize;
        if i >= self.live.len() || !self.live[i] {
            return Err(AnnError::IdOutOfRange {
                id: i,
                n: self.graph.len(),
            });
        }
        self.deleted.insert(id);
        // If the entry just got tombstoned, refresh it so search keeps working.
        if self.entry == Some(id) {
            self.entry = self.pick_entry();
        }
        Ok(())
    }

    /// Consolidate all pending deletions: bridge edges over deleted nodes,
    /// re-prune affected nodes back to degree `R`, then release the deleted slots
    /// to the free list. Returns the number of slots reclaimed.
    ///
    /// Implements the paper's delete-consolidation: for every live node `p` that
    /// links to a deleted node `d`, edge `p → d` is replaced by `p → N_out(d)`
    /// (minus deleted targets and `p` itself); `N_out(p)` is then RobustPruned.
    pub fn consolidate(&mut self) -> usize {
        if self.deleted.is_empty() {
            return 0;
        }
        let deleted = self.deleted.clone();

        // Repair every live, non-deleted node whose out-list touches a deletion.
        for p in 0..self.graph.len() as u32 {
            let pi = p as usize;
            if !self.live[pi] || deleted.contains(&p) {
                continue;
            }
            let touches = self.graph[pi].iter().any(|n| deleted.contains(n));
            if !touches {
                continue;
            }

            // Build the bridged candidate set: keep live non-deleted neighbours,
            // and for each deleted neighbour splice in its (live) out-neighbours.
            let mut cand_ids: HashSet<u32> = HashSet::new();
            let old: Vec<u32> = self.graph[pi].clone();
            for n in old {
                if deleted.contains(&n) {
                    for &nn in &self.graph[n as usize] {
                        if nn != p && !deleted.contains(&nn) {
                            cand_ids.insert(nn);
                        }
                    }
                } else if n != p {
                    cand_ids.insert(n);
                }
            }

            let mut cand: Vec<(u32, f32)> = cand_ids
                .iter()
                .map(|&v| (v, self.dist_nodes(p, v)))
                .collect();
            self.graph[pi] = self.robust_prune(p, &mut cand);
        }

        // Release deleted slots.
        let mut reclaimed = 0usize;
        for d in &deleted {
            let di = *d as usize;
            self.live[di] = false;
            self.graph[di].clear();
            self.free_slots.push(*d);
            reclaimed += 1;
        }
        self.deleted.clear();
        // Entry may have referenced a now-dead slot.
        self.entry = self.pick_entry();
        reclaimed
    }

    /// Greedy best-first traversal collecting the visited candidate set,
    /// ascending by distance to `query`, bounded to `l`. Tombstoned and dead
    /// slots are skipped. `exclude` keeps a specific slot out of the result
    /// (used during insert so `p` is not its own neighbour).
    fn greedy_collect(&self, query: &[f32], l: usize, exclude: Option<u32>) -> Vec<(u32, f32)> {
        let Some(start) = self.pick_entry_excluding(exclude) else {
            return Vec::new();
        };

        let mut frontier: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
        let mut result: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
        let mut visited: HashSet<u32> = HashSet::new();
        let mut consumed: HashSet<u32> = HashSet::new();

        let d_start = self.dist_query(query, start);
        frontier.push(Reverse((OrdF32(d_start), start)));
        result.push((OrdF32(d_start), start));
        visited.insert(start);

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
                // Traverse through deleted nodes (they keep the graph connected)
                // but never *add* a deleted/dead node to the candidate result.
                if !visited.insert(nbr) {
                    continue;
                }
                let nbr_i = nbr as usize;
                let admissible = self.live[nbr_i] && !self.deleted.contains(&nbr);
                let d = self.dist_query(query, nbr);
                // Always allow expansion via frontier for connectivity.
                frontier.push(Reverse((OrdF32(d), nbr)));
                if !admissible {
                    continue;
                }
                let worst = result.peek().map_or(f32::INFINITY, |(OrdF32(w), _)| *w);
                if result.len() < l {
                    result.push((OrdF32(d), nbr));
                } else if d < worst {
                    result.pop();
                    result.push((OrdF32(d), nbr));
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

    /// Like [`Self::pick_entry`] but never returns the excluded slot (so a
    /// freshly-allocated insert target is not used as its own search seed).
    fn pick_entry_excluding(&self, exclude: Option<u32>) -> Option<u32> {
        if let Some(e) = self.entry
            && self.contains(e)
            && Some(e) != exclude
        {
            return Some(e);
        }
        (0..self.graph.len() as u32).find(|&s| self.contains(s) && Some(s) != exclude)
    }

    /// RobustPrune (DiskANN Algorithm 2): from `candidates`, pick ≤ `R`
    /// out-neighbours of `p` with the `α`-relaxed greedy rule. Deleted / dead
    /// targets are filtered out so consolidation never reintroduces a hole.
    fn robust_prune(&self, p: u32, candidates: &mut Vec<(u32, f32)>) -> Vec<u32> {
        candidates.retain(|&(v, _)| v != p && self.live[v as usize] && !self.deleted.contains(&v));
        if candidates.is_empty() {
            return Vec::new();
        }
        candidates.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let r = self.cfg.degree_r;
        let alpha = self.cfg.alpha;
        let mut result: Vec<u32> = Vec::with_capacity(r);
        let mut alive: Vec<bool> = vec![true; candidates.len()];
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

    /// Approximate top-`k` nearest neighbours of `query` (ascending by L2²).
    /// Deleted points are never returned.
    ///
    /// # Errors
    /// - [`AnnError::InvalidK`] if `k == 0`.
    /// - [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`AnnError::IndexEmpty`] if there are no searchable points.
    pub fn search(&self, query: &[f32], k: usize) -> AnnResult<Vec<(u32, f32)>> {
        if k == 0 {
            return Err(AnnError::InvalidK { k, n: self.len() });
        }
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if self.is_empty() {
            return Err(AnnError::IndexEmpty);
        }
        let l = self.cfg.search_l.max(k);
        let mut res = self.greedy_collect(query, l, None);
        let actual_k = k.min(res.len());
        res.truncate(actual_k);
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn cfg() -> FreshDiskAnnConfig {
        FreshDiskAnnConfig {
            degree_r: 8,
            search_l: 16,
            alpha: 1.2,
        }
    }

    fn rand_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0_f32; n * dim];
        rng.fill_normal(&mut v);
        v
    }

    fn brute_topk(data: &[f32], n: usize, dim: usize, q: &[f32], k: usize) -> Vec<usize> {
        let mut d: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = &data[i * dim..(i + 1) * dim];
                let dd: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                (i, dd)
            })
            .collect();
        d.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        d.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn new_rejects_zero_dim() {
        assert!(matches!(
            FreshDiskAnnIndex::new(0, cfg()),
            Err(AnnError::InvalidVectorDim { dim: 0 })
        ));
    }

    #[test]
    fn new_rejects_bad_cfg() {
        let bad = FreshDiskAnnConfig {
            degree_r: 0,
            search_l: 8,
            alpha: 1.0,
        };
        assert!(FreshDiskAnnIndex::new(4, bad).is_err());
        let bad2 = FreshDiskAnnConfig {
            degree_r: 4,
            search_l: 8,
            alpha: 0.5,
        };
        assert!(FreshDiskAnnIndex::new(4, bad2).is_err());
    }

    #[test]
    fn insert_then_self_find() {
        let dim = 8;
        let n = 80;
        let data = rand_data(n, dim, 1);
        let mut idx = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        for i in 0..n {
            idx.insert(&data[i * dim..(i + 1) * dim])
                .expect("insert ok");
        }
        assert_eq!(idx.len(), n);
        // Approximate graph search recovers each vector as its own top-k=5
        // neighbour for the overwhelming majority of nodes (an incrementally
        // built navigable graph does not guarantee rank-1 self-retrieval for
        // every single node, but recall must be near-perfect at k=5).
        let k = 5;
        let mut self_found = 0usize;
        for i in 0..n {
            let q = &data[i * dim..(i + 1) * dim];
            let res = idx.search(q, k).expect("search ok");
            assert!(!res.is_empty());
            if res.iter().any(|&(id, d)| id as usize == i && d < 1e-5) {
                self_found += 1;
            }
        }
        let recall = self_found as f32 / n as f32;
        assert!(recall >= 0.95, "self-find recall@{k} = {recall:.3} < 0.95");
    }

    #[test]
    fn recall_against_brute_force() {
        let dim = 12;
        let n = 200;
        let data = rand_data(n, dim, 2);
        let mut idx = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        for i in 0..n {
            idx.insert(&data[i * dim..(i + 1) * dim])
                .expect("insert ok");
        }
        let k = 10;
        let queries = rand_data(20, dim, 999);
        let mut hits = 0usize;
        for qi in 0..20 {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let gt: HashSet<usize> = brute_topk(&data, n, dim, q, k).into_iter().collect();
            let res = idx.search(q, k).expect("search ok");
            hits += res
                .iter()
                .filter(|(id, _)| gt.contains(&(*id as usize)))
                .count();
        }
        let recall = hits as f32 / (20 * k) as f32;
        assert!(recall >= 0.8, "recall={recall:.3} < 0.8");
    }

    #[test]
    fn deleted_points_not_returned() {
        let dim = 6;
        let n = 60;
        let data = rand_data(n, dim, 3);
        let mut idx = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        for i in 0..n {
            idx.insert(&data[i * dim..(i + 1) * dim])
                .expect("insert ok");
        }
        // Delete node 10 and query with its own vector.
        idx.delete(10).expect("delete ok");
        assert_eq!(idx.pending_deletions(), 1);
        assert_eq!(idx.len(), n - 1);
        let q = &data[10 * dim..11 * dim];
        let res = idx.search(q, 5).expect("search ok");
        assert!(
            res.iter().all(|(id, _)| *id != 10),
            "deleted id 10 returned: {res:?}"
        );
    }

    #[test]
    fn consolidate_reclaims_and_keeps_recall() {
        let dim = 10;
        let n = 150;
        let data = rand_data(n, dim, 4);
        let mut idx = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        for i in 0..n {
            idx.insert(&data[i * dim..(i + 1) * dim])
                .expect("insert ok");
        }
        // Delete 30 of them.
        let to_delete: Vec<u32> = (0..30u32).map(|i| i * 5).collect();
        for &d in &to_delete {
            idx.delete(d).expect("delete ok");
        }
        let reclaimed = idx.consolidate();
        assert_eq!(reclaimed, to_delete.len());
        assert_eq!(idx.pending_deletions(), 0);
        assert_eq!(idx.len(), n - to_delete.len());

        // Surviving set.
        let alive: Vec<usize> = (0..n)
            .filter(|i| !to_delete.contains(&(*i as u32)))
            .collect();
        // Recall over surviving ground truth.
        let k = 8;
        let queries = rand_data(15, dim, 555);
        let mut hits = 0usize;
        for qi in 0..15 {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut d: Vec<(usize, f32)> = alive
                .iter()
                .map(|&i| {
                    let v = &data[i * dim..(i + 1) * dim];
                    let dd: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i, dd)
                })
                .collect();
            d.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let gt: HashSet<usize> = d.into_iter().take(k).map(|(i, _)| i).collect();
            let res = idx.search(q, k).expect("search ok");
            // No deleted ids may appear.
            assert!(res.iter().all(|(id, _)| !to_delete.contains(id)));
            hits += res
                .iter()
                .filter(|(id, _)| gt.contains(&(*id as usize)))
                .count();
        }
        let recall = hits as f32 / (15 * k) as f32;
        assert!(
            recall >= 0.75,
            "post-consolidation recall={recall:.3} < 0.75"
        );
    }

    #[test]
    fn insert_reuses_freed_slot() {
        let dim = 4;
        let data = rand_data(20, dim, 5);
        let mut idx = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        for i in 0..20 {
            idx.insert(&data[i * dim..(i + 1) * dim])
                .expect("insert ok");
        }
        let cap_before = idx.capacity();
        idx.delete(7).expect("delete ok");
        idx.delete(8).expect("delete ok");
        idx.consolidate();
        // Two slots free now; two inserts should reuse them (no capacity growth).
        let extra = rand_data(2, dim, 6);
        let id_a = idx.insert(&extra[0..dim]).expect("insert ok");
        let id_b = idx.insert(&extra[dim..2 * dim]).expect("insert ok");
        assert_eq!(
            idx.capacity(),
            cap_before,
            "capacity grew despite free slots"
        );
        assert!([7u32, 8].contains(&id_a));
        assert!([7u32, 8].contains(&id_b));
        assert_ne!(id_a, id_b);
        assert!(idx.contains(id_a));
    }

    #[test]
    fn search_after_delete_all_then_reinsert() {
        let dim = 5;
        let data = rand_data(30, dim, 7);
        let mut idx = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        for i in 0..30 {
            idx.insert(&data[i * dim..(i + 1) * dim])
                .expect("insert ok");
        }
        for i in 0..30u32 {
            idx.delete(i).expect("delete ok");
        }
        assert!(idx.is_empty());
        assert!(matches!(
            idx.search(&data[0..dim], 1),
            Err(AnnError::IndexEmpty)
        ));
        let reclaimed = idx.consolidate();
        assert_eq!(reclaimed, 30, "all 30 deleted slots should be reclaimed");
        assert!(idx.is_empty());
        // Reinsert one and find it.
        let nid = idx.insert(&data[0..dim]).expect("insert ok");
        let res = idx.search(&data[0..dim], 1).expect("search ok");
        assert_eq!(res[0].0, nid);
        assert!(res[0].1 < 1e-5);
    }

    #[test]
    fn delete_missing_id_errors() {
        let dim = 4;
        let mut idx = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        idx.insert(&[1.0, 2.0, 3.0, 4.0]).expect("insert ok");
        assert!(matches!(idx.delete(99), Err(AnnError::IdOutOfRange { .. })));
    }

    #[test]
    fn dim_mismatch_on_insert_and_search() {
        let dim = 4;
        let mut idx = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        assert!(matches!(
            idx.insert(&[1.0, 2.0, 3.0]),
            Err(AnnError::DimensionMismatch { .. })
        ));
        idx.insert(&[1.0, 2.0, 3.0, 4.0]).expect("insert ok");
        assert!(matches!(
            idx.search(&[1.0, 2.0], 1),
            Err(AnnError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            idx.search(&[1.0, 2.0, 3.0, 4.0], 0),
            Err(AnnError::InvalidK { .. })
        ));
    }

    #[test]
    fn consolidate_noop_when_no_deletions() {
        let dim = 4;
        let data = rand_data(10, dim, 8);
        let mut idx = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        for i in 0..10 {
            idx.insert(&data[i * dim..(i + 1) * dim])
                .expect("insert ok");
        }
        assert_eq!(idx.consolidate(), 0);
    }

    #[test]
    fn degree_bounded_after_inserts_and_consolidation() {
        let dim = 8;
        let n = 120;
        let data = rand_data(n, dim, 9);
        let mut idx = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        for i in 0..n {
            idx.insert(&data[i * dim..(i + 1) * dim])
                .expect("insert ok");
        }
        for i in 0..20u32 {
            idx.delete(i * 6).expect("delete ok");
        }
        idx.consolidate();
        for slot in 0..idx.capacity() as u32 {
            if idx.contains(slot) {
                assert!(
                    idx.neighbors(slot).len() <= cfg().degree_r,
                    "slot {slot} exceeds degree R"
                );
            }
        }
    }

    #[test]
    fn deterministic_same_inserts() {
        let dim = 6;
        let n = 50;
        let data = rand_data(n, dim, 11);
        let mut a = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        let mut b = FreshDiskAnnIndex::new(dim, cfg()).expect("cfg valid");
        for i in 0..n {
            a.insert(&data[i * dim..(i + 1) * dim]).expect("insert ok");
            b.insert(&data[i * dim..(i + 1) * dim]).expect("insert ok");
        }
        for slot in 0..n as u32 {
            assert_eq!(a.neighbors(slot), b.neighbors(slot), "slot {slot} differs");
        }
    }
}
