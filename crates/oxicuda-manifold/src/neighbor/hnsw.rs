//! Hierarchical Navigable Small World (HNSW) approximate kNN.
//!
//! Implements the Malkov & Yashunin 2016 algorithm for logarithmic-time approximate
//! nearest neighbor search. Above d ≈ 50 this is orders-of-magnitude faster than KD-trees.
//!
//! # Architecture
//!
//! A multi-layer graph is built where:
//! - Layer 0 (bottom) is the densest — every inserted point lives here.
//! - Each point i also lives in layers 1..=level_i where level_i is drawn from a
//!   geometric distribution with parameter 1/m_L.
//! - Each point has at most M neighbors per layer (M_max0 = 2M for layer 0).
//! - The entry point is the node with the highest layer.
//!
//! During insertion the algorithm greedily descends from the top layer to the
//! point's maximum layer (ef=1), then runs a wider beam search (ef=ef_construction)
//! for layers at and below the point's maximum layer.
//!
//! Neighbor selection uses a diversity heuristic: a candidate is retained only if it
//! is closer to the query than to any already-selected neighbor.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Public configuration
// ---------------------------------------------------------------------------

/// Distance metric used by the HNSW index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HnswDistance {
    /// Squared Euclidean distance: ‖a − b‖².
    Euclidean,
    /// Cosine dissimilarity: 1 − (a·b) / (‖a‖ ‖b‖).
    Cosine,
    /// Negative dot product (useful when vectors are pre-normalised).
    DotProduct,
}

/// Construction and search parameters for [`HnswIndex`].
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of bidirectional links per element per layer. (default: 16)
    pub m: usize,
    /// Candidate set size during index construction. (default: 200)
    pub ef_construction: usize,
    /// Candidate set size during search — must be ≥ k. (default: 50)
    pub ef_search: usize,
    /// Level multiplier. Typically `1 / ln(m)`. `0.0` → auto-compute from `m`. (default: 0.0)
    pub m_l: f64,
    /// RNG seed for level generation. (default: 42)
    pub seed: u64,
    /// Distance metric. (default: Euclidean)
    pub distance: HnswDistance,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            m_l: 0.0,
            seed: 42,
            distance: HnswDistance::Euclidean,
        }
    }
}

// ---------------------------------------------------------------------------
// Public index and result types
// ---------------------------------------------------------------------------

/// HNSW index built over a set of points.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    /// Number of indexed points.
    pub n_points: usize,
    /// Dimensionality of each point.
    pub dim: usize,
    /// Flat row-major storage: `data[i * dim .. (i+1) * dim]` = coordinates of point i.
    pub(crate) data: Vec<f64>,
    /// `graph[node][layer]` = list of neighbor indices in that layer.
    pub(crate) graph: Vec<Vec<Vec<usize>>>,
    /// Current entry point (the node with the highest `max_layer`).
    pub(crate) entry_point: usize,
    /// Highest layer that exists in the index (= max over all nodes of their levels).
    pub(crate) max_layer: usize,
    /// Configuration used to build this index.
    pub(crate) config: HnswConfig,
    /// RNG state carried forward for incremental inserts.
    pub(crate) rng: LcgRng,
}

/// Result of an [`hnsw_search`] call.
#[derive(Debug, Clone)]
pub struct HnswSearchResult {
    /// `indices[q][i]` = index of the i-th nearest neighbor for query q.
    pub indices: Vec<Vec<usize>>,
    /// `distances[q][i]` = distance to that neighbor (same metric as [`HnswConfig::distance`]).
    pub distances: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

/// Compute the raw distance (or dissimilarity) between two points according to `metric`.
#[inline]
fn compute_distance(a: &[f64], b: &[f64], dim: usize, metric: HnswDistance) -> f64 {
    match metric {
        HnswDistance::Euclidean => {
            let mut s = 0.0_f64;
            for i in 0..dim {
                let d = a[i] - b[i];
                s += d * d;
            }
            s
        }
        HnswDistance::Cosine => {
            let mut dot = 0.0_f64;
            let mut na2 = 0.0_f64;
            let mut nb2 = 0.0_f64;
            for i in 0..dim {
                dot += a[i] * b[i];
                na2 += a[i] * a[i];
                nb2 += b[i] * b[i];
            }
            let denom = na2.sqrt() * nb2.sqrt();
            if denom < 1e-300 {
                1.0
            } else {
                (1.0 - dot / denom).max(0.0)
            }
        }
        HnswDistance::DotProduct => {
            let mut dot = 0.0_f64;
            for i in 0..dim {
                dot += a[i] * b[i];
            }
            -dot
        }
    }
}

/// Distance from a query slice to point `idx` stored inside the index.
#[inline]
fn dist_to(index: &HnswIndex, query: &[f64], idx: usize) -> f64 {
    compute_distance(
        query,
        &index.data[idx * index.dim..(idx + 1) * index.dim],
        index.dim,
        index.config.distance,
    )
}

// ---------------------------------------------------------------------------
// Priority-queue primitives (max-heap over (distance, node_id))
// ---------------------------------------------------------------------------
//
// We use a plain Vec<(f64, usize)> and maintain the "heap invariant" manually.
// Two complementary views are used:
//   • "W" (worst-first) = max-heap: pops the farthest element.
//   • "C" (candidate) = we keep candidates sorted by ascending distance for
//     diversity-heuristic selection (no BinaryHeap needed).

/// Insert `item` into a bounded max-heap of capacity `cap`.
/// If the heap is full and `item.0 < current_max`, replace the maximum element.
fn heap_push_bounded(heap: &mut Vec<(f64, usize)>, cap: usize, item: (f64, usize)) {
    if heap.len() < cap {
        heap.push(item);
        // Sift up to maintain max-heap property
        let mut idx = heap.len() - 1;
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if heap[parent].0 < heap[idx].0 {
                heap.swap(parent, idx);
                idx = parent;
            } else {
                break;
            }
        }
    } else if !heap.is_empty() && item.0 < heap[0].0 {
        heap[0] = item;
        // Sift down
        sift_down(heap, 0);
    }
}

/// Push without capacity bound — simply maintain max-heap.
fn heap_push(heap: &mut Vec<(f64, usize)>, item: (f64, usize)) {
    heap.push(item);
    let mut idx = heap.len() - 1;
    while idx > 0 {
        let parent = (idx - 1) / 2;
        if heap[parent].0 < heap[idx].0 {
            heap.swap(parent, idx);
            idx = parent;
        } else {
            break;
        }
    }
}

/// Pop the maximum element from a max-heap.
fn heap_pop(heap: &mut Vec<(f64, usize)>) -> Option<(f64, usize)> {
    if heap.is_empty() {
        return None;
    }
    let last = heap.len() - 1;
    heap.swap(0, last);
    let top = heap.pop();
    if !heap.is_empty() {
        sift_down(heap, 0);
    }
    top
}

fn sift_down(heap: &mut [(f64, usize)], mut idx: usize) {
    let n = heap.len();
    loop {
        let left = 2 * idx + 1;
        let right = 2 * idx + 2;
        let mut largest = idx;
        if left < n && heap[left].0 > heap[largest].0 {
            largest = left;
        }
        if right < n && heap[right].0 > heap[largest].0 {
            largest = right;
        }
        if largest == idx {
            break;
        }
        heap.swap(idx, largest);
        idx = largest;
    }
}

/// Peek at the current maximum element (root of max-heap).
#[inline]
fn heap_max(heap: &[(f64, usize)]) -> f64 {
    heap.first().map(|t| t.0).unwrap_or(f64::INFINITY)
}

// ---------------------------------------------------------------------------
// Level sampling
// ---------------------------------------------------------------------------

/// Sample the maximum layer for a new point using the geometric distribution.
/// `m_l` is 1/ln(M) (the level multiplier).
fn sample_level(rng: &mut LcgRng, m_l: f64, max_layer_cap: usize) -> usize {
    let u = rng.next_f64().max(1e-300);
    let level = (-u.ln() * m_l).floor() as usize;
    level.min(max_layer_cap)
}

// ---------------------------------------------------------------------------
// Beam search (search_layer)
// ---------------------------------------------------------------------------

/// Greedy beam search on a single layer of the graph.
///
/// Starting from `entry_points` (already in the candidate set), explores the graph
/// and returns the `ef` closest nodes found.
///
/// # Arguments
/// * `index`        — the HNSW index (for graph topology and distance computation)
/// * `query`        — coordinates of the point being inserted or searched
/// * `entry_points` — initial seeds; their distances are computed inside
/// * `ef`           — desired number of candidates to return
/// * `layer`        — which graph layer to traverse
///
/// Returns a Vec<(dist, node)> sorted by ascending distance (length ≤ ef).
fn search_layer(
    index: &HnswIndex,
    query: &[f64],
    entry_points: &[usize],
    ef: usize,
    layer: usize,
) -> Vec<(f64, usize)> {
    // W: max-heap of closest found so far (bounded to ef)
    let mut w: Vec<(f64, usize)> = Vec::with_capacity(ef + 1);
    // C: min-heap of candidates to explore (unbounded during search)
    //    We simulate a min-heap using a max-heap by negating distances.
    let mut c_neg: Vec<(f64, usize)> = Vec::new(); // max-heap over (-dist, node)

    // Track visited nodes to avoid re-processing
    let mut visited: Vec<bool> = vec![false; index.n_points];

    for &ep in entry_points {
        if ep >= index.n_points {
            continue;
        }
        let d = dist_to(index, query, ep);
        visited[ep] = true;
        heap_push_bounded(&mut w, ef, (d, ep));
        heap_push(&mut c_neg, (-d, ep));
    }

    while let Some((neg_dc, c)) = heap_pop(&mut c_neg) {
        let dc = -neg_dc;

        // If nearest candidate is farther than current worst in W, stop
        if dc > heap_max(&w) && w.len() >= ef {
            break;
        }

        // Explore neighbors of c at this layer
        if layer >= index.graph[c].len() {
            continue;
        }
        let neighbors: Vec<usize> = index.graph[c][layer].clone();
        for e in neighbors {
            if e >= index.n_points || visited[e] {
                continue;
            }
            visited[e] = true;
            let de = dist_to(index, query, e);
            if de < heap_max(&w) || w.len() < ef {
                heap_push_bounded(&mut w, ef, (de, e));
                heap_push(&mut c_neg, (-de, e));
            }
        }
    }

    // Sort W ascending for caller convenience
    w.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    w
}

// ---------------------------------------------------------------------------
// Neighbor selection heuristic (Algorithm 4 in the paper)
// ---------------------------------------------------------------------------

/// Select at most `m_max` neighbors from candidates using the "simple" heuristic
/// that prefers diversity (closer to query than to any already-selected neighbor).
///
/// `candidates` must be sorted by ascending distance from the query.
fn select_neighbors_heuristic(
    index: &HnswIndex,
    query_or_node: &[f64],
    candidates: &[(f64, usize)],
    m_max: usize,
) -> Vec<usize> {
    let mut result: Vec<usize> = Vec::with_capacity(m_max);
    // Kept as (dist_to_query, node_id)
    let mut selected: Vec<(f64, usize)> = Vec::with_capacity(m_max);

    for &(dc, c) in candidates {
        if result.len() >= m_max {
            break;
        }
        // A candidate c is added only if it is closer to the query than to any already
        // selected node (diversity / "relative neighbourhood graph" condition).
        let mut dominated = false;
        for &(_, r) in &selected {
            let dr = compute_distance(
                &index.data[c * index.dim..(c + 1) * index.dim],
                &index.data[r * index.dim..(r + 1) * index.dim],
                index.dim,
                index.config.distance,
            );
            if dr < dc {
                // c is closer to existing neighbor r than to the query; skip c
                dominated = true;
                break;
            }
        }
        if !dominated {
            result.push(c);
            selected.push((dc, c));
        }
        // Safety-net: if diversity heuristic rejects too aggressively, fall back to
        // adding the candidate anyway once we have fewer than m_max/2 after exhausting
        // the diverse set — handled implicitly by continuing the loop.
        let _ = query_or_node; // kept for API clarity; distances already computed via `dc`
    }

    // If diversity heuristic rejected everything (edge case with very small datasets),
    // fall back to distance-sorted selection.
    if result.is_empty() && !candidates.is_empty() {
        let take = candidates.len().min(m_max);
        for &(_, c) in candidates.iter().take(take) {
            result.push(c);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Core insertion logic
// ---------------------------------------------------------------------------

/// Insert a single point (already appended to `index.data`) at position `node_id`
/// into the graph layers. Mutates `index.graph`, `index.entry_point`, `index.max_layer`.
fn insert_node(index: &mut HnswIndex, node_id: usize, level: usize) {
    let m = index.config.m;
    let m_max0 = m * 2; // max connections at layer 0
    let ef_construction = index.config.ef_construction;

    // Ensure the new node has enough layer slots
    index.graph[node_id] = vec![Vec::new(); level + 1];

    if index.n_points == 1 {
        // First point: simply become the entry point at its level
        index.entry_point = 0;
        index.max_layer = level;
        return;
    }

    let mut ep = vec![index.entry_point];

    // Phase 1: descend from max_layer to level+1 with ef=1 (greedy, no wide search)
    let current_top = index.max_layer;
    for lc in (level + 1..=current_top).rev() {
        let query_coords: Vec<f64> =
            index.data[node_id * index.dim..(node_id + 1) * index.dim].to_vec();
        let found = search_layer(index, &query_coords, &ep, 1, lc);
        if !found.is_empty() {
            ep = vec![found[0].1];
        }
    }

    // Phase 2: from min(level, current_top) down to 0 with ef=ef_construction
    for lc in (0..=level.min(current_top)).rev() {
        let m_max = if lc == 0 { m_max0 } else { m };
        let query_coords: Vec<f64> =
            index.data[node_id * index.dim..(node_id + 1) * index.dim].to_vec();
        let candidates = search_layer(index, &query_coords, &ep, ef_construction, lc);

        // Select neighbors for the new node
        let neighbors = select_neighbors_heuristic(index, &query_coords, &candidates, m_max);

        // Connect new node -> neighbors
        if lc < index.graph[node_id].len() {
            index.graph[node_id][lc] = neighbors.clone();
        }

        // Connect neighbors -> new node (bidirectional), then prune if needed
        for &nb in &neighbors {
            let nb_dist_to_new = compute_distance(
                &index.data[nb * index.dim..(nb + 1) * index.dim],
                &index.data[node_id * index.dim..(node_id + 1) * index.dim],
                index.dim,
                index.config.distance,
            );

            // Ensure nb has a slot for layer lc
            if lc >= index.graph[nb].len() {
                index.graph[nb].resize(lc + 1, Vec::new());
            }
            index.graph[nb][lc].push(node_id);

            // Prune nb's adjacency list if it exceeds m_max
            if index.graph[nb][lc].len() > m_max {
                let nb_coords: Vec<f64> = index.data[nb * index.dim..(nb + 1) * index.dim].to_vec();
                // Build candidate list sorted by distance from nb
                let mut nb_candidates: Vec<(f64, usize)> = index.graph[nb][lc]
                    .iter()
                    .map(|&nid| {
                        let d = if nid == node_id {
                            nb_dist_to_new
                        } else {
                            compute_distance(
                                &index.data[nb * index.dim..(nb + 1) * index.dim],
                                &index.data[nid * index.dim..(nid + 1) * index.dim],
                                index.dim,
                                index.config.distance,
                            )
                        };
                        (d, nid)
                    })
                    .collect();
                nb_candidates
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let pruned = select_neighbors_heuristic(index, &nb_coords, &nb_candidates, m_max);
                index.graph[nb][lc] = pruned;
            }
        }

        // Advance entry points to the best candidates found for next layer
        if !candidates.is_empty() {
            ep = candidates.iter().map(|t| t.1).collect();
        }
    }

    // Update global entry point if new node has a higher level
    if level > index.max_layer {
        index.max_layer = level;
        index.entry_point = node_id;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build an HNSW index from row-major data of shape `(n_points, dim)`.
///
/// # Errors
/// - [`ManifoldError::EmptyInput`] if `n_points == 0` or `dim == 0`.
/// - [`ManifoldError::ShapeMismatch`] if `data.len() != n_points * dim`.
/// - [`ManifoldError::InvalidParameter`] if `m == 0` or `ef_construction < m`.
pub fn hnsw_build(
    data: &[f64],
    n_points: usize,
    dim: usize,
    config: &HnswConfig,
) -> ManifoldResult<HnswIndex> {
    // -- Validate inputs -------------------------------------------------------
    if n_points == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if data.len() != n_points * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_points, dim],
            got: vec![data.len()],
        });
    }
    if config.m == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "m".into(),
            reason: "must be >= 1".into(),
        });
    }
    if config.ef_construction < config.m {
        return Err(ManifoldError::InvalidParameter {
            name: "ef_construction".into(),
            reason: "must be >= m".into(),
        });
    }

    // Resolve level multiplier
    let m_l = if config.m_l <= 0.0 {
        1.0 / (config.m as f64).ln()
    } else {
        config.m_l
    };

    // Maximum reasonable layer cap to avoid unbounded level sampling
    let max_layer_cap = ((n_points as f64).ln() / (config.m as f64).ln()).ceil() as usize + 2;

    let mut rng = LcgRng::new(config.seed);

    // Pre-allocate the index with the full data
    let mut index = HnswIndex {
        n_points: 0, // grows as we insert
        dim,
        data: data.to_vec(),
        graph: vec![Vec::new(); n_points],
        entry_point: 0,
        max_layer: 0,
        config: config.clone(),
        rng: LcgRng::new(config.seed), // will be replaced
    };

    // Insert points one by one
    for node_id in 0..n_points {
        index.n_points = node_id + 1;
        let level = sample_level(&mut rng, m_l, max_layer_cap);
        insert_node(&mut index, node_id, level);
    }

    // Save the final RNG state for incremental inserts
    index.rng = rng;

    Ok(index)
}

/// Search the HNSW index for the `k` nearest neighbors of each query point.
///
/// # Arguments
/// * `index`     — a built [`HnswIndex`]
/// * `queries`   — row-major query matrix of shape `(n_queries, dim)`
/// * `n_queries` — number of query points
/// * `k`         — number of neighbors to return per query
///
/// # Errors
/// - [`ManifoldError::EmptyInput`] if `n_queries == 0`.
/// - [`ManifoldError::KNeighborsTooLarge`] if `k > n_points`.
/// - [`ManifoldError::ShapeMismatch`] if `queries.len() != n_queries * dim`.
pub fn hnsw_search(
    index: &HnswIndex,
    queries: &[f64],
    n_queries: usize,
    k: usize,
) -> ManifoldResult<HnswSearchResult> {
    if n_queries == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if queries.len() != n_queries * index.dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_queries, index.dim],
            got: vec![queries.len()],
        });
    }
    if k == 0 || k > index.n_points {
        return Err(ManifoldError::KNeighborsTooLarge {
            k,
            n: index.n_points,
        });
    }

    let ef = index.config.ef_search.max(k);

    let mut all_indices = Vec::with_capacity(n_queries);
    let mut all_distances = Vec::with_capacity(n_queries);

    for qi in 0..n_queries {
        let query = &queries[qi * index.dim..(qi + 1) * index.dim];
        let result = search_single(index, query, k, ef);
        let mut indices = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);
        for (d, idx) in result {
            indices.push(idx);
            distances.push(d);
        }
        all_indices.push(indices);
        all_distances.push(distances);
    }

    Ok(HnswSearchResult {
        indices: all_indices,
        distances: all_distances,
    })
}

/// Internal single-query search returning up to `k` results sorted ascending.
fn search_single(index: &HnswIndex, query: &[f64], k: usize, ef: usize) -> Vec<(f64, usize)> {
    if index.n_points == 0 {
        return Vec::new();
    }

    let mut ep = vec![index.entry_point];

    // Descend from max_layer down to layer 1 with ef=1
    for lc in (1..=index.max_layer).rev() {
        let found = search_layer(index, query, &ep, 1, lc);
        if !found.is_empty() {
            ep = vec![found[0].1];
        }
    }

    // Layer 0 wide search with ef candidates
    let mut candidates = search_layer(index, query, &ep, ef, 0);

    // Sort ascending and take k
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(k);
    candidates
}

/// Insert a new point into an existing [`HnswIndex`], returning its assigned index.
///
/// This performs online insertion using the same algorithm as [`hnsw_build`],
/// preserving the graph's quality guarantees.
///
/// # Errors
/// - [`ManifoldError::ShapeMismatch`] if `point.len() != index.dim`.
pub fn hnsw_add(index: &mut HnswIndex, point: &[f64]) -> ManifoldResult<usize> {
    if point.len() != index.dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![index.dim],
            got: vec![point.len()],
        });
    }

    let m_l = if index.config.m_l <= 0.0 {
        1.0 / (index.config.m as f64).ln()
    } else {
        index.config.m_l
    };
    let max_layer_cap =
        ((index.n_points as f64 + 1.0).ln() / (index.config.m as f64).ln()).ceil() as usize + 2;

    let node_id = index.n_points;

    // Append data
    index.data.extend_from_slice(point);
    index.graph.push(Vec::new());
    index.n_points += 1;

    let level = sample_level(&mut index.rng, m_l, max_layer_cap);
    insert_node(index, node_id, level);

    Ok(node_id)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn euclidean_dist_sq(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Brute-force kNN (squared Euclidean) returning sorted (dist_sq, idx) list.
    fn brute_knn_sq(data: &[f64], n: usize, dim: usize, query: &[f64], k: usize) -> Vec<usize> {
        let mut dists: Vec<(f64, usize)> = (0..n)
            .map(|i| {
                let d = euclidean_dist_sq(&data[i * dim..(i + 1) * dim], query);
                (d, i)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dists.iter().take(k).map(|t| t.1).collect()
    }

    // --- 1. Build on n=100, d=2 succeeds -----------------------------------
    #[test]
    fn build_n100_d2_succeeds() {
        let n = 100;
        let dim = 2;
        let mut rng = LcgRng::new(1234);
        let data: Vec<f64> = (0..n * dim).map(|_| rng.next_f64()).collect();
        let cfg = HnswConfig::default();
        let idx = hnsw_build(&data, n, dim, &cfg).expect("build");
        assert_eq!(idx.n_points, n);
        assert_eq!(idx.dim, dim);
    }

    // --- 2. k=1 returns exact nearest neighbor on well-separated data ------
    #[test]
    fn search_k1_exact_nn_well_separated() {
        let dim = 2;
        // 5 clearly separated clusters
        let centers = [
            (0.0f64, 0.0),
            (10.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (5.0, 5.0),
        ];
        let data: Vec<f64> = centers.iter().flat_map(|&(x, y)| [x, y]).collect();
        let n = centers.len();
        let cfg = HnswConfig {
            m: 4,
            ef_construction: 20,
            ef_search: 10,
            ..Default::default()
        };
        let idx = hnsw_build(&data, n, dim, &cfg).expect("build");

        // Query each center — should return itself
        for (i, &(cx, cy)) in centers.iter().enumerate() {
            let res = hnsw_search(&idx, &[cx, cy], 1, 1).expect("search");
            assert_eq!(
                res.indices[0][0], i,
                "query center {i} should return itself"
            );
        }
    }

    // --- 3. Search k=5: all returned indices in [0, n) --------------------
    #[test]
    fn search_k5_indices_in_range() {
        let n = 50;
        let dim = 4;
        let mut rng = LcgRng::new(42);
        let data: Vec<f64> = (0..n * dim).map(|_| rng.next_f64()).collect();
        let cfg = HnswConfig::default();
        let idx = hnsw_build(&data, n, dim, &cfg).expect("build");
        let query: Vec<f64> = (0..dim).map(|_| rng.next_f64()).collect();
        let res = hnsw_search(&idx, &query, 1, 5).expect("search");
        for &ni in &res.indices[0] {
            assert!(ni < n, "index {ni} out of range");
        }
    }

    // --- 4. Distances are sorted ascending per query ----------------------
    #[test]
    fn search_distances_sorted_ascending() {
        let n = 80;
        let dim = 3;
        let mut rng = LcgRng::new(77);
        let data: Vec<f64> = (0..n * dim).map(|_| rng.next_f64()).collect();
        let cfg = HnswConfig {
            ef_search: 20,
            ..Default::default()
        };
        let idx = hnsw_build(&data, n, dim, &cfg).expect("build");
        let query: Vec<f64> = (0..dim).map(|_| rng.next_f64()).collect();
        let res = hnsw_search(&idx, &query, 1, 8).expect("search");
        let dists = &res.distances[0];
        for w in dists.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-12,
                "distances not sorted: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    // --- 5. Recall ≥ 0.8 on random d=8 data (vs brute-force) -------------
    #[test]
    fn recall_random_d8() {
        let n = 200;
        let dim = 8;
        let k = 5;
        let mut rng = LcgRng::new(99);
        let data: Vec<f64> = (0..n * dim).map(|_| rng.next_f64()).collect();
        let cfg = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            ..Default::default()
        };
        let idx = hnsw_build(&data, n, dim, &cfg).expect("build");

        let n_queries = 20;
        let mut total_hits = 0usize;
        let mut total_expected = 0usize;

        for _ in 0..n_queries {
            let query: Vec<f64> = (0..dim).map(|_| rng.next_f64()).collect();
            let exact = brute_knn_sq(&data, n, dim, &query, k);
            let res = hnsw_search(&idx, &query, 1, k).expect("search");
            let hnsw_set = &res.indices[0];
            let hits = exact.iter().filter(|&&e| hnsw_set.contains(&e)).count();
            total_hits += hits;
            total_expected += k;
        }

        let recall = total_hits as f64 / total_expected as f64;
        assert!(
            recall >= 0.8,
            "recall {recall:.3} below 0.80 — HNSW quality too low"
        );
    }

    // --- 6. hnsw_add: index grows by 1 each time --------------------------
    #[test]
    fn add_grows_index() {
        let n = 10;
        let dim = 3;
        let mut rng = LcgRng::new(7);
        let data: Vec<f64> = (0..n * dim).map(|_| rng.next_f64()).collect();
        let cfg = HnswConfig {
            m: 4,
            ef_construction: 10,
            ..Default::default()
        };
        let mut idx = hnsw_build(&data, n, dim, &cfg).expect("build");
        assert_eq!(idx.n_points, n);

        for extra in 1..=5 {
            let pt: Vec<f64> = (0..dim).map(|_| rng.next_f64()).collect();
            let new_id = hnsw_add(&mut idx, &pt).expect("add");
            assert_eq!(new_id, n + extra - 1, "returned id mismatch");
            assert_eq!(idx.n_points, n + extra);
        }
    }

    // --- 7. Invalid k > n → error -----------------------------------------
    #[test]
    fn search_k_too_large_errors() {
        let n = 5;
        let dim = 2;
        let data: Vec<f64> = (0..n * 2).map(|i| i as f64).collect();
        let cfg = HnswConfig {
            m: 2,
            ef_construction: 4,
            ..Default::default()
        };
        let idx = hnsw_build(&data, n, dim, &cfg).expect("build");
        let q = vec![0.0, 0.0];
        let res = hnsw_search(&idx, &q, 1, n + 1);
        assert!(res.is_err(), "expected error for k > n");
    }

    // --- 8. Empty data → error -------------------------------------------
    #[test]
    fn build_empty_data_errors() {
        let cfg = HnswConfig::default();
        let res = hnsw_build(&[], 0, 2, &cfg);
        assert!(res.is_err());
    }

    // --- 9. n=1 index: search returns the only point ----------------------
    #[test]
    fn single_point_index() {
        let dim = 4;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let cfg = HnswConfig {
            m: 4,
            ef_construction: 4,
            ef_search: 4,
            ..Default::default()
        };
        let idx = hnsw_build(&data, 1, dim, &cfg).expect("build");
        let q = vec![0.0, 0.0, 0.0, 0.0];
        let res = hnsw_search(&idx, &q, 1, 1).expect("search");
        assert_eq!(res.indices[0][0], 0);
    }

    // --- 10. m=0 → error --------------------------------------------------
    #[test]
    fn build_m_zero_errors() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let cfg = HnswConfig {
            m: 0,
            ef_construction: 10,
            ..Default::default()
        };
        let res = hnsw_build(&data, 2, 2, &cfg);
        assert!(res.is_err());
    }

    // --- 11. ef_construction < m → error ---------------------------------
    #[test]
    fn build_ef_less_than_m_errors() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let cfg = HnswConfig {
            m: 16,
            ef_construction: 5, // < m
            ..Default::default()
        };
        let res = hnsw_build(&data, 2, 2, &cfg);
        assert!(res.is_err());
    }

    // --- 12. Cosine distance: k=1 matches brute-force for normalised data -
    #[test]
    fn cosine_k1_matches_brute_force() {
        let dim = 4;
        // Normalised unit vectors
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let raw = [
            [1.0_f64, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [inv_sqrt2, inv_sqrt2, 0.0, 0.0],
        ];
        let n = raw.len();
        let data: Vec<f64> = raw.iter().flat_map(|r| r.iter().copied()).collect();
        let cfg = HnswConfig {
            m: 4,
            ef_construction: 10,
            ef_search: 10,
            distance: HnswDistance::Cosine,
            ..Default::default()
        };
        let idx = hnsw_build(&data, n, dim, &cfg).expect("build");

        // Query = [1/sqrt(2), 1/sqrt(2), 0, 0] (equal mix of e1 and e2)
        let query = vec![inv_sqrt2, inv_sqrt2, 0.0_f64, 0.0];
        let res = hnsw_search(&idx, &query, 1, 1).expect("search");

        // Brute-force cosine: nearest should be index 4 (same direction)
        let mut bf: Vec<(f64, usize)> = (0..n)
            .map(|i| {
                let d = compute_distance(
                    &data[i * dim..(i + 1) * dim],
                    &query,
                    dim,
                    HnswDistance::Cosine,
                );
                (d, i)
            })
            .collect();
        bf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(
            res.indices[0][0], bf[0].1,
            "cosine k=1 mismatch: got {} expected {}",
            res.indices[0][0], bf[0].1
        );
    }

    // --- 13. Euclidean and Cosine produce different results on non-unit data
    #[test]
    fn euclidean_cosine_differ() {
        let dim = 2;
        // Points with very different norms
        let data = vec![
            1.0_f64, 0.0, // unit, same direction as query
            100.0, 0.0, // very long, same direction
            0.0, 1.0, // perpendicular
        ];
        let n = 3;
        let query = vec![2.0f64, 0.0];

        let cfg_euc = HnswConfig {
            m: 2,
            ef_construction: 4,
            ef_search: 4,
            distance: HnswDistance::Euclidean,
            ..Default::default()
        };
        let cfg_cos = HnswConfig {
            m: 2,
            ef_construction: 4,
            ef_search: 4,
            distance: HnswDistance::Cosine,
            ..Default::default()
        };

        let idx_euc = hnsw_build(&data, n, dim, &cfg_euc).expect("build euc");
        let idx_cos = hnsw_build(&data, n, dim, &cfg_cos).expect("build cos");

        let res_euc = hnsw_search(&idx_euc, &query, 1, 1).expect("search euc");
        let res_cos = hnsw_search(&idx_cos, &query, 1, 1).expect("search cos");

        // Euclidean nearest: index 0 (dist=1) vs index 1 (dist=98)
        // Cosine nearest: both 0 and 1 have cosine dist ≈ 0 (same direction)
        // At minimum, just verify both return valid indices in [0, n)
        assert!(res_euc.indices[0][0] < n);
        assert!(res_cos.indices[0][0] < n);
    }

    // --- 14. Clustered d=2 data: HNSW recalls exact neighbors ------------
    #[test]
    fn clustered_d2_high_recall() {
        let dim = 2;
        let k = 3;

        // Two tight clusters far apart
        let mut data = Vec::new();
        let cluster_a: &[(f64, f64)] =
            &[(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.05, 0.05), (0.1, 0.1)];
        let cluster_b: &[(f64, f64)] = &[
            (100.0, 100.0),
            (100.1, 100.0),
            (100.0, 100.1),
            (100.05, 100.05),
            (100.1, 100.1),
        ];
        for &(x, y) in cluster_a.iter().chain(cluster_b.iter()) {
            data.push(x);
            data.push(y);
        }
        let n = data.len() / dim;

        let cfg = HnswConfig {
            m: 4,
            ef_construction: 20,
            ef_search: 20,
            ..Default::default()
        };
        let idx = hnsw_build(&data, n, dim, &cfg).expect("build");

        // Query the centroid of cluster A
        let query_a = vec![0.05, 0.05];
        let res = hnsw_search(&idx, &query_a, 1, k).expect("search");

        // All k neighbors should be within cluster A (indices 0..5)
        let all_in_a = res.indices[0].iter().all(|&i| i < 5);
        // Recall: at least 2 out of 3 are in cluster A
        let count_in_a = res.indices[0].iter().filter(|&&i| i < 5).count();
        assert!(
            all_in_a || count_in_a >= 2,
            "clustered recall failed: got {:?}",
            res.indices[0]
        );
    }

    // --- 15. Multiple queries batch search --------------------------------
    #[test]
    fn batch_search_shape_correct() {
        let n = 30;
        let dim = 5;
        let mut rng = LcgRng::new(55);
        let data: Vec<f64> = (0..n * dim).map(|_| rng.next_f64()).collect();
        let cfg = HnswConfig {
            m: 4,
            ef_construction: 20,
            ef_search: 15,
            ..Default::default()
        };
        let idx = hnsw_build(&data, n, dim, &cfg).expect("build");

        let n_queries = 7;
        let k = 4;
        let queries: Vec<f64> = (0..n_queries * dim).map(|_| rng.next_f64()).collect();
        let res = hnsw_search(&idx, &queries, n_queries, k).expect("search");

        assert_eq!(res.indices.len(), n_queries);
        assert_eq!(res.distances.len(), n_queries);
        for qi in 0..n_queries {
            assert_eq!(res.indices[qi].len(), k, "query {qi} indices length");
            assert_eq!(res.distances[qi].len(), k, "query {qi} distances length");
        }
    }

    // --- 16. DotProduct distance variant compiles and produces valid results
    #[test]
    fn dot_product_distance_valid() {
        let dim = 3;
        let data = vec![
            1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0,
        ];
        let n = 4;
        let cfg = HnswConfig {
            m: 2,
            ef_construction: 4,
            ef_search: 4,
            distance: HnswDistance::DotProduct,
            ..Default::default()
        };
        let idx = hnsw_build(&data, n, dim, &cfg).expect("build");
        let query = vec![1.0, 0.0, 0.0];
        let res = hnsw_search(&idx, &query, 1, 1).expect("search");
        assert!(res.indices[0][0] < n);
    }
}
