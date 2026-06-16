//! Vamana — in-memory core of the DiskANN graph index.
//!
//! Reference: Subramanya, Devvrit, Kadekodi, Krishnaswamy, Simhadri, *"DiskANN:
//! Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"*,
//! NeurIPS 2019. We implement the in-memory Vamana graph construction (the
//! algorithmic core of DiskANN); SSD residency (block layout, IO pipelining,
//! second-pass index merging) is intentionally left for a future module.
//!
//! Build outline (paraphrased from the paper):
//! 1. Initialise every node with an empty out-neighbour list.
//! 2. Pick a `start_node`. For determinism we use **node 0** (documented).
//! 3. For each point `p` in deterministic insertion order:
//!    - Run `greedy_search(p, L)` over the partially-built graph to collect a
//!      candidate set `V` (the visited set of size up to `L`).
//!    - `out_neighbours(p) ← robust_prune(p, V, α, R)`.
//!    - For every `q` chosen as a neighbour of `p`, add the back-edge
//!      `q → p`; if `|out_neighbours(q)| > R` we re-prune `q`'s neighbour
//!      list with `robust_prune(q, neighbours(q) ∪ {p}, α, R)`.
//!
//! RobustPrune (Algorithm 2 of the paper):
//! ```text
//! V ← sorted ascending by dist(p, ·)
//! result ← []
//! while V not empty and |result| < R:
//!     v* ← V.pop_front()                       # closest remaining
//!     result.append(v*)
//!     V.retain(|v'| α · dist(v*, v') > dist(p, v'))   # drop dominated
//! ```
//! Geometrically: we keep `v*` and discard every `v'` that lies inside the ball
//! of radius `dist(p, v')/α` around `v*`, i.e. every `v'` that is closer to
//! `v*` than to `p` (slack by `α`).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::error::{AnnError, AnnResult};

/// Totally-ordered `f32` wrapper for use inside binary heaps. `NaN` sinks to
/// the "largest" bucket so it does not poison ordering.
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

/// Vamana build / search configuration.
#[derive(Debug, Clone)]
pub struct VamanaConfig {
    /// Maximum out-degree `R`. Must be `>= 1`.
    pub degree_r: usize,
    /// Candidate-list size `L` used by greedy search during the build. Must
    /// be `>= 1`.
    pub search_l: usize,
    /// RobustPrune slack `α`. Typically `1.0..=1.3`; must be `>= 1.0`.
    pub alpha: f32,
}

/// In-memory Vamana graph.
pub struct VamanaIndex {
    /// Flat `n × dim` row-major point storage.
    points: Vec<f32>,
    /// `graph[p]` is the (bounded) out-neighbour list of node `p`.
    graph: Vec<Vec<u32>>,
    /// Deterministic start node used to seed every greedy search.
    start_node: usize,
    /// Cached configuration.
    cfg: VamanaConfig,
    /// Vector dimensionality (fixed at build time, `0` before).
    dim: usize,
}

impl VamanaIndex {
    /// Create an empty index from a validated configuration.
    ///
    /// # Errors
    /// [`AnnError::Internal`] if any of `degree_r`, `search_l` is `0`, or
    /// `alpha < 1.0` (or non-finite).
    pub fn new(cfg: VamanaConfig) -> AnnResult<Self> {
        if cfg.degree_r == 0 {
            return Err(AnnError::Internal {
                msg: "degree_r must be >= 1".to_string(),
            });
        }
        if cfg.search_l == 0 {
            return Err(AnnError::Internal {
                msg: "search_l must be >= 1".to_string(),
            });
        }
        if !(cfg.alpha.is_finite()) || cfg.alpha < 1.0 {
            return Err(AnnError::Internal {
                msg: format!("alpha must be a finite value >= 1.0, got {}", cfg.alpha),
            });
        }
        Ok(Self {
            points: Vec::new(),
            graph: Vec::new(),
            start_node: 0,
            cfg,
            dim: 0,
        })
    }

    /// Number of indexed points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// `true` when no points have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Configured vector dimensionality after a successful [`Self::build`].
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
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

    /// Deterministic start node id (node 0 by construction).
    #[must_use]
    pub fn start_node(&self) -> u32 {
        self.start_node as u32
    }

    /// Borrow point `id`.
    fn point(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.points[s..s + self.dim]
    }

    /// Squared-L2 distance between `query` and node `id` (callers validate
    /// lengths).
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

    /// Build the in-memory Vamana graph over `data` (row-major `n × dim`).
    ///
    /// Uses node 0 as the deterministic `start_node` and inserts points in
    /// ascending id order. For every newly-inserted point `p`:
    /// 1. `greedy_search(p, L)` returns up to `L` visited candidates.
    /// 2. `out_neighbours(p) ← robust_prune(p, V, α, R)`.
    /// 3. For each chosen `q`: add `q → p`; if `|out(q)| > R`,
    ///    re-prune `q`.
    ///
    /// # Errors
    /// - [`AnnError::EmptyInput`] if `n == 0`.
    /// - [`AnnError::InvalidVectorDim`] if `dim == 0`.
    /// - [`AnnError::DimensionMismatch`] if `data.len() != n * dim`.
    pub fn build(&mut self, data: &[f32], n: usize, dim: usize) -> AnnResult<()> {
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

        // Initialise storage with every point loaded up-front. We then iterate
        // build-style, but distances reference the final coordinate array.
        self.points = data.to_vec();
        self.graph = vec![Vec::new(); n];
        self.start_node = 0;
        self.dim = dim;

        // First-pass insertion in ascending id order.
        // (DiskANN performs two passes with different α to reach degree R;
        // a single pass with α=1 gives the navigable backbone, while values
        // > 1 add "long-range" shortcuts. We honour the configured α as a
        // single-pass build per the in-memory contract.)
        for p in 0..n as u32 {
            // Empty / single-point graphs trivially have no candidates yet.
            if p == 0 {
                continue;
            }

            // greedy_search uses the partially-built graph (only neighbours of
            // nodes < p are populated). `p`'s vector is already in `points`,
            // so we can query with it directly.
            let q_vec: Vec<f32> = self.point(p).to_vec();
            let mut visited = self.greedy_search_internal(&q_vec, self.cfg.search_l, Some(p))?;

            // Run RobustPrune to choose out-neighbours of `p`.
            let pruned = self.robust_prune_internal(p, &mut visited)?;
            self.graph[p as usize] = pruned.clone();

            // Back-edges, with re-pruning when the neighbour's list overflows.
            for q in pruned {
                let qi = q as usize;
                if !self.graph[qi].contains(&p) {
                    self.graph[qi].push(p);
                }
                if self.graph[qi].len() > self.cfg.degree_r {
                    // Re-prune q from its (current ∪ {p}) neighbour set.
                    let mut cand: Vec<(u32, f32)> = self.graph[qi]
                        .iter()
                        .filter(|&&v| v != q)
                        .map(|&v| (v, self.dist_nodes(q, v)))
                        .collect();
                    let new_nbrs = self.robust_prune_internal(q, &mut cand)?;
                    self.graph[qi] = new_nbrs;
                }
            }
        }

        Ok(())
    }

    /// Greedy navigable best-first search returning the candidate list
    /// (visited set), ascending by distance to `query`, bounded to `l` items.
    ///
    /// # Errors
    /// - [`AnnError::IndexEmpty`] if the graph contains no points.
    /// - [`AnnError::InvalidK`] if `l == 0`.
    /// - [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    pub fn greedy_search(&self, query: &[f32], l: usize) -> AnnResult<Vec<(u32, f32)>> {
        if self.graph.is_empty() {
            return Err(AnnError::IndexEmpty);
        }
        if l == 0 {
            return Err(AnnError::InvalidK {
                k: l,
                n: self.graph.len(),
            });
        }
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        self.greedy_search_internal(query, l, None)
    }

    /// Internal greedy search that may exclude a specific node id (used during
    /// the build to keep `p` out of its own visited set).
    fn greedy_search_internal(
        &self,
        query: &[f32],
        l: usize,
        exclude: Option<u32>,
    ) -> AnnResult<Vec<(u32, f32)>> {
        if self.graph.is_empty() {
            return Ok(Vec::new());
        }

        // Start from the deterministic medoid replacement (node 0). If node 0
        // is the excluded one and another node exists, start from any other
        // present id.
        let n = self.graph.len() as u32;
        let start = match exclude {
            Some(p) if p == self.start_node as u32 => {
                // Find the first id != p that has at least an empty slot.
                let mut s: u32 = 0;
                while s < n && s == p {
                    s += 1;
                }
                if s == n {
                    return Ok(Vec::new());
                }
                s
            }
            _ => self.start_node as u32,
        };

        // frontier: min-heap by distance.
        let mut frontier: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();
        // result: max-heap by distance, bounded to L.
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
            // Early termination: once the closest unexpanded frontier point is
            // worse than the current `L`-th best, no improvement is possible.
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
        Ok(out)
    }

    /// Approximate top-`k` nearest neighbours of `query`, ascending by L2².
    ///
    /// Internally runs [`Self::greedy_search`] with candidate list size
    /// `max(L, k)` and truncates to `k`.
    ///
    /// # Errors
    /// - [`AnnError::InvalidK`] if `k == 0`.
    /// - [`AnnError::DimensionMismatch`] if `query.len() != dim`.
    /// - [`AnnError::IndexEmpty`] if the graph is empty.
    pub fn search(&self, query: &[f32], k: usize) -> AnnResult<Vec<(u32, f32)>> {
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
        let l = self.cfg.search_l.max(k);
        let mut res = self.greedy_search_internal(query, l, None)?;
        let actual_k = k.min(res.len());
        res.truncate(actual_k);
        Ok(res)
    }

    /// RobustPrune (Algorithm 2 of the DiskANN paper): from candidate set
    /// `candidates`, choose up to `R` out-neighbours for node `p` using the
    /// `α`-relaxed greedy rule.
    ///
    /// Order of operations:
    /// 1. Drop `p` itself from `candidates`.
    /// 2. Sort `candidates` ascending by `dist(p, v)`.
    /// 3. Repeat: pop the closest remaining `v*`, append to result, and remove
    ///    every `v'` whose `α · dist(v*, v') ≤ dist(p, v')` (i.e. `v'` is
    ///    closer to `v*` than to `p`, modulo the slack).
    ///
    /// # Errors
    /// Currently infallible (always returns `Ok`) — wrapped in `AnnResult` so
    /// callers do not have to special-case the API.
    pub fn robust_prune(&self, p: u32, candidates: &mut Vec<(u32, f32)>) -> AnnResult<Vec<u32>> {
        self.robust_prune_internal(p, candidates)
    }

    fn robust_prune_internal(
        &self,
        p: u32,
        candidates: &mut Vec<(u32, f32)>,
    ) -> AnnResult<Vec<u32>> {
        // Drop self.
        candidates.retain(|&(v, _)| v != p);
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        // Ascending by distance to p.
        candidates.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let r = self.cfg.degree_r;
        let alpha = self.cfg.alpha;
        let mut result: Vec<u32> = Vec::with_capacity(r);

        // Process candidates in distance order using an index, removing dominated
        // entries lazily via an "alive" mask. (sort + linear scan + retain.)
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

            // Drop every still-alive v' with α * dist(v*, v') <= dist(p, v').
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

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> VamanaConfig {
        VamanaConfig {
            degree_r: 4,
            search_l: 8,
            alpha: 1.2,
        }
    }

    /// Five well-separated 2-D clusters.
    fn well_separated_5() -> (Vec<f32>, usize, usize) {
        let centres = [
            [0.0_f32, 0.0],
            [100.0, 0.0],
            [0.0, 100.0],
            [-100.0, 0.0],
            [50.0, -100.0],
        ];
        let mut data = Vec::new();
        for c in centres.iter() {
            data.extend_from_slice(c);
        }
        (data, centres.len(), 2)
    }

    #[test]
    fn build_sets_len() {
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        let dim = 2;
        let data: Vec<f32> = (0..10).flat_map(|i| [i as f32, -i as f32]).collect();
        idx.build(&data, 10, dim).expect("build with valid data");
        assert_eq!(idx.len(), 10);
        assert!(!idx.is_empty());
        assert_eq!(idx.dim(), dim);
    }

    #[test]
    fn every_node_respects_degree_r() {
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        let dim = 2;
        let data: Vec<f32> = (0..15).flat_map(|i| [i as f32, (i * 2) as f32]).collect();
        idx.build(&data, 15, dim).expect("build with valid data");
        for id in 0..15 {
            let nb = idx.neighbors(id as u32);
            assert!(
                nb.len() <= idx.cfg.degree_r,
                "node {id} has {} > R={} neighbours",
                nb.len(),
                idx.cfg.degree_r
            );
        }
    }

    #[test]
    fn greedy_search_returns_sorted_bounded() {
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        let dim = 2;
        let data: Vec<f32> = (0..20).flat_map(|i| [i as f32, (10 - i) as f32]).collect();
        idx.build(&data, 20, dim).expect("build with valid data");
        let q = vec![5.0_f32, 5.0];
        let res = idx
            .greedy_search(&q, 6)
            .expect("greedy_search with valid parameters");
        assert!(res.len() <= 6);
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1, "not ascending: {res:?}");
        }
    }

    #[test]
    fn search_returns_sorted_and_bounded() {
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        let dim = 2;
        let data: Vec<f32> = (0..20).flat_map(|i| [i as f32, (10 - i) as f32]).collect();
        idx.build(&data, 20, dim).expect("build with valid data");
        let q = vec![5.0_f32, 5.0];
        let res = idx.search(&q, 3).expect("search with valid parameters");
        assert!(res.len() <= 3);
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn exact_nn_on_well_separated_clusters() {
        let (data, n, dim) = well_separated_5();
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        idx.build(&data, n, dim).expect("build with valid data");
        for i in 0..n {
            let q = &data[i * dim..(i + 1) * dim];
            let res = idx.search(q, 1).expect("search with valid parameters");
            assert_eq!(res.len(), 1);
            assert_eq!(res[0].0 as usize, i, "i={i} res={res:?}");
            assert!(res[0].1 < 1e-6);
        }
    }

    #[test]
    fn robust_prune_respects_degree_cap() {
        let mut idx = VamanaIndex::new(VamanaConfig {
            degree_r: 3,
            search_l: 8,
            alpha: 1.2,
        })
        .expect("config is valid");
        let dim = 2;
        let data: Vec<f32> = (0..10).flat_map(|i| [i as f32, 0.0]).collect();
        idx.build(&data, 10, dim).expect("build with valid data");
        let mut cand: Vec<(u32, f32)> = (1u32..10).map(|v| (v, idx.dist_nodes(0, v))).collect();
        let pruned = idx
            .robust_prune(0, &mut cand)
            .expect("robust_prune is infallible");
        assert!(pruned.len() <= 3, "pruned={pruned:?}");
    }

    #[test]
    fn robust_prune_respects_alpha_dominance() {
        let mut idx = VamanaIndex::new(VamanaConfig {
            degree_r: 5,
            search_l: 8,
            alpha: 1.2,
        })
        .expect("config is valid");
        let dim = 2;
        // 6 collinear points where the prune rule has plenty of dominated pairs.
        let data: Vec<f32> = (0..6).flat_map(|i| [i as f32, 0.0]).collect();
        idx.build(&data, 6, dim).expect("build with valid data");
        let mut cand: Vec<(u32, f32)> = (1u32..6).map(|v| (v, idx.dist_nodes(0, v))).collect();
        let pruned = idx
            .robust_prune(0, &mut cand)
            .expect("robust_prune is infallible");
        let alpha = idx.cfg.alpha;
        // For every retained pair (v, v'), with v earlier than v' in the
        // prune output, the rule guarantees α · dist(v, v') > dist(p, v').
        let p = 0u32;
        for i in 0..pruned.len() {
            for j in (i + 1)..pruned.len() {
                let v = pruned[i];
                let vp = pruned[j];
                let d_vv = idx.dist_nodes(v, vp);
                let d_pvp = idx.dist_query(idx.point(p), vp);
                assert!(
                    alpha * d_vv > d_pvp,
                    "i={i} j={j} v={v} v'={vp} α·d(v,v')={} d(p,v')={}",
                    alpha * d_vv,
                    d_pvp
                );
            }
        }
    }

    #[test]
    fn robust_prune_empty_input() {
        let idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        let mut cand: Vec<(u32, f32)> = Vec::new();
        let pruned = idx
            .robust_prune(0, &mut cand)
            .expect("robust_prune is infallible");
        assert!(pruned.is_empty());
    }

    #[test]
    fn deterministic_build_no_rng() {
        let dim = 2;
        let data: Vec<f32> = (0..12).flat_map(|i| [i as f32, (3 * i) as f32]).collect();
        let mut a = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        let mut b = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        a.build(&data, 12, dim).expect("build with valid data");
        b.build(&data, 12, dim).expect("build with valid data");
        for id in 0..12u32 {
            assert_eq!(a.neighbors(id), b.neighbors(id), "id={id}");
        }
        let q = vec![3.0_f32, 9.0];
        assert_eq!(
            a.search(&q, 3).expect("search with valid parameters"),
            b.search(&q, 3).expect("search with valid parameters")
        );
    }

    #[test]
    fn err_degree_zero() {
        let cfg = VamanaConfig {
            degree_r: 0,
            search_l: 8,
            alpha: 1.2,
        };
        assert!(matches!(
            VamanaIndex::new(cfg),
            Err(AnnError::Internal { .. })
        ));
    }

    #[test]
    fn err_search_l_zero() {
        let cfg = VamanaConfig {
            degree_r: 4,
            search_l: 0,
            alpha: 1.2,
        };
        assert!(matches!(
            VamanaIndex::new(cfg),
            Err(AnnError::Internal { .. })
        ));
    }

    #[test]
    fn err_alpha_below_one() {
        let cfg = VamanaConfig {
            degree_r: 4,
            search_l: 8,
            alpha: 0.5,
        };
        assert!(matches!(
            VamanaIndex::new(cfg),
            Err(AnnError::Internal { .. })
        ));
    }

    #[test]
    fn err_build_data_length_mismatch() {
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        let r = idx.build(&[0.0_f32, 1.0, 2.0], 2, 2);
        assert!(matches!(r, Err(AnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_search_query_wrong_length() {
        let (data, n, dim) = well_separated_5();
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        idx.build(&data, n, dim).expect("build with valid data");
        let bad = vec![0.0_f32; 3];
        let r = idx.search(&bad, 1);
        assert!(matches!(r, Err(AnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_k_zero() {
        let (data, n, dim) = well_separated_5();
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        idx.build(&data, n, dim).expect("build with valid data");
        let q = vec![0.0_f32; dim];
        let r = idx.search(&q, 0);
        assert!(matches!(r, Err(AnnError::InvalidK { k: 0, .. })));
    }

    #[test]
    fn err_n_zero_and_dim_zero() {
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        assert!(matches!(idx.build(&[], 0, 2), Err(AnnError::EmptyInput)));
        assert!(matches!(
            idx.build(&[], 1, 0),
            Err(AnnError::InvalidVectorDim { dim: 0 })
        ));
    }

    #[test]
    fn single_point_index_returns_that_point() {
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        idx.build(&[3.0_f32, 4.0], 1, 2)
            .expect("build with valid data");
        let res = idx
            .search(&[3.0_f32, 4.0], 1)
            .expect("search with valid parameters");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 0);
        assert!(res[0].1.abs() < 1e-6);
    }

    #[test]
    fn connectivity_every_node_reachable_from_start() {
        let (data, n, dim) = well_separated_5();
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        idx.build(&data, n, dim).expect("build with valid data");
        // BFS from start_node in the directed graph -> every node must be visited.
        let start = idx.start_node() as usize;
        let mut visited = vec![false; n];
        let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        visited[start] = true;
        q.push_back(start);
        while let Some(u) = q.pop_front() {
            for &v in idx.neighbors(u as u32) {
                let vi = v as usize;
                if !visited[vi] {
                    visited[vi] = true;
                    q.push_back(vi);
                }
            }
        }
        assert!(visited.iter().all(|&b| b), "visited={visited:?}");
    }

    #[test]
    fn duplicate_points_handled() {
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        let dim = 2;
        let data = vec![
            0.0_f32, 0.0, // 0
            0.0, 0.0, // 1 (dup of 0)
            5.0, 5.0, // 2
            5.0, 5.0, // 3 (dup of 2)
        ];
        idx.build(&data, 4, dim).expect("build with valid data");
        assert_eq!(idx.len(), 4);
        let res = idx
            .search(&[0.0_f32, 0.0], 2)
            .expect("search with valid parameters");
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|&(_, d)| d.abs() < 1e-6));
        // Both top-2 should be the duplicated pair {0, 1}.
        let mut ids: Vec<u32> = res.iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0u32, 1]);
    }

    #[test]
    fn greedy_search_k_greater_than_n_returns_n() {
        let (data, n, dim) = well_separated_5();
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        idx.build(&data, n, dim).expect("build with valid data");
        let q = vec![0.0_f32; dim];
        let res = idx
            .greedy_search(&q, 100)
            .expect("greedy_search with valid parameters");
        assert_eq!(res.len(), n);
    }

    #[test]
    fn search_k_greater_than_n_returns_n() {
        let (data, n, dim) = well_separated_5();
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        idx.build(&data, n, dim).expect("build with valid data");
        let q = vec![0.0_f32; dim];
        let res = idx.search(&q, 100).expect("search with valid parameters");
        assert_eq!(res.len(), n);
    }

    #[test]
    fn err_greedy_search_l_zero() {
        let (data, n, dim) = well_separated_5();
        let mut idx = VamanaIndex::new(small_cfg()).expect("small_cfg is valid");
        idx.build(&data, n, dim).expect("build with valid data");
        let q = vec![0.0_f32; dim];
        let r = idx.greedy_search(&q, 0);
        assert!(matches!(r, Err(AnnError::InvalidK { k: 0, .. })));
    }

    #[test]
    fn err_alpha_non_finite() {
        let cfg = VamanaConfig {
            degree_r: 4,
            search_l: 8,
            alpha: f32::NAN,
        };
        assert!(matches!(
            VamanaIndex::new(cfg),
            Err(AnnError::Internal { .. })
        ));
    }
}
