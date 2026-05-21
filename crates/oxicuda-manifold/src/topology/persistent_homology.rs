//! Persistent Homology and Mapper algorithm for Topological Data Analysis (TDA).
//!
//! # Algorithms
//!
//! ## Vietoris-Rips Persistent Homology (H0 and H1)
//!
//! Given a finite metric space (point cloud) and an increasing filtration parameter ε,
//! the Vietoris-Rips complex VR(X, ε) includes:
//! - 0-simplices (vertices): all points.
//! - 1-simplices (edges): pairs (i,j) with d(i,j) ≤ ε.
//! - 2-simplices (triangles): triples (i,j,k) with d(i,j), d(i,k), d(j,k) all ≤ ε.
//!
//! **H0 (connected components):** tracked via union-find during edge insertion sorted by weight.
//! Each merge event gives a persistence pair (birth, death).
//!
//! **H1 (loops/cycles):** computed via boundary matrix column reduction
//! (Edelsbrunner-Letscher-Zomorodian 2002). The boundary matrix has:
//!
//! - rows = edges (1-simplices)
//! - columns = triangles (2-simplices)
//! - each column stores the three boundary edges of the triangle.
//! - column reduction pairs 1-simplices with 2-simplices:
//!   - Unpaired edge → H1 birth (with death = ∞, i.e., essential 1-cycle)
//!   - Paired (edge, triangle) → H1 birth at edge filtration, death at triangle filtration
//!
//! ## Mapper
//!
//! A graph-based topological summary of the data:
//! 1. Project each point onto a scalar filter function f (distance to centroid).
//! 2. Cover the range [f_min, f_max] with overlapping intervals.
//! 3. For each interval, cluster the preimage using k-means.
//! 4. Create a node for each (interval, cluster) pair.
//! 5. Connect two nodes if they share at least one data point.
//!
//! # References
//! - Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). *Topological Persistence and Simplification.*
//! - Singh, G., Mémoli, F., Carlsson, G. (2007). *Topological Methods for the Analysis of High Dimensional Data.*

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single birth-death pair in a persistence diagram.
///
/// Represents the topological feature that is born at filtration value `birth`
/// and dies at filtration value `death`. If `death == f64::INFINITY`, the feature
/// is *essential* (never dies within the filtration).
#[derive(Debug, Clone, PartialEq)]
pub struct PersistencePair {
    /// Filtration value at which this topological feature is born.
    pub birth: f64,
    /// Filtration value at which this topological feature dies.
    /// `f64::INFINITY` if the feature is essential (never dies).
    pub death: f64,
    /// Homological dimension: 0 = connected component, 1 = loop/cycle.
    pub dimension: usize,
}

impl PersistencePair {
    /// Persistence (death − birth) of this pair.
    /// Returns `f64::INFINITY` for essential pairs.
    #[must_use]
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Returns `true` if this is an essential (infinite-lived) feature.
    #[must_use]
    pub fn is_essential(&self) -> bool {
        self.death.is_infinite()
    }
}

/// A persistence diagram containing all birth-death pairs for H0 and H1.
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    /// All persistence pairs (both H0 and H1).
    pub pairs: Vec<PersistencePair>,
    /// Number of essential H0 classes (infinite-lived connected components).
    pub betti_0: usize,
    /// Number of essential H1 classes (infinite-lived loops).
    pub betti_1: usize,
    /// The maximum filtration radius used.
    pub max_filtration: f64,
}

impl PersistenceDiagram {
    /// Filter pairs by homological dimension.
    #[must_use]
    pub fn pairs_in_dimension(&self, dim: usize) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| p.dimension == dim).collect()
    }

    /// Total number of finite (non-essential) pairs.
    #[must_use]
    pub fn n_finite_pairs(&self) -> usize {
        self.pairs.iter().filter(|p| !p.is_essential()).count()
    }
}

/// Configuration for the Vietoris-Rips filtration.
#[derive(Debug, Clone)]
pub struct VietorisRipsConfig {
    /// Maximum filtration radius ε_max.
    ///
    /// If `0.0`, the algorithm auto-selects `diameter / 4.0` where `diameter`
    /// is the maximum pairwise distance in the dataset.
    pub max_radius: f64,

    /// Maximum homological dimension to compute.
    ///
    /// - `0` → H0 only (connected components, very fast).
    /// - `1` → H0 + H1 (loops, requires boundary matrix reduction).
    pub max_dimension: usize,

    /// Expected number of points, used for sanity-checking against data slice length.
    pub n_points: usize,
}

impl Default for VietorisRipsConfig {
    fn default() -> Self {
        Self {
            max_radius: 0.0,
            max_dimension: 1,
            n_points: 0,
        }
    }
}

/// Configuration for the Mapper algorithm.
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// Number of overlapping cover intervals along the filter function axis.
    pub n_intervals: usize,
    /// Fractional overlap between consecutive intervals (0 < overlap < 1).
    pub overlap: f64,
    /// Number of clusters to find within each interval.
    pub n_clusters: usize,
    /// Seed for the k-means random number generator.
    pub seed: u64,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            n_intervals: 10,
            overlap: 0.5,
            n_clusters: 2,
            seed: 42,
        }
    }
}

/// A node in the Mapper graph, corresponding to one (interval, cluster) pair.
#[derive(Debug, Clone)]
pub struct MapperNode {
    /// Index of the cover interval that this node belongs to.
    pub interval_idx: usize,
    /// Index of the cluster within that interval.
    pub cluster_idx: usize,
    /// Indices of the original data points belonging to this node.
    pub point_indices: Vec<usize>,
    /// Number of points in this node (`point_indices.len()`).
    pub size: usize,
}

/// A topological graph produced by the Mapper algorithm.
#[derive(Debug, Clone)]
pub struct MapperGraph {
    /// All nodes in the Mapper graph.
    pub nodes: Vec<MapperNode>,
    /// Edges between nodes as pairs of node indices `(u, v)` with `u < v`.
    pub edges: Vec<(usize, usize)>,
}

impl MapperGraph {
    /// Number of nodes in the graph.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph.
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Adjacency list representation (node index → list of neighbor node indices).
    #[must_use]
    pub fn adjacency_list(&self) -> Vec<Vec<usize>> {
        let n = self.nodes.len();
        let mut adj = vec![Vec::<usize>::new(); n];
        for &(u, v) in &self.edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        adj
    }
}

// ---------------------------------------------------------------------------
// Union-Find (Disjoint-Set Union) for H0
// ---------------------------------------------------------------------------

/// Union-Find data structure with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    /// Filtration value at which this component was born.
    birth: Vec<f64>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            birth: vec![0.0; n],
        }
    }

    /// Find root of the component containing `x` with path compression.
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the components of `x` and `y`. Returns `Some((killed_root, surviving_root))` if they
    /// were distinct (a merge happened), or `None` if they were already connected.
    fn union(&mut self, x: usize, y: usize) -> Option<(usize, usize)> {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return None;
        }
        // Union by rank: higher-rank root absorbs the lower-rank root.
        // Ties: ry absorbs rx (older birth = lower index wins as canonical).
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
            Some((rx, ry))
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
            Some((ry, rx))
        } else {
            self.parent[rx] = ry;
            self.rank[ry] += 1;
            Some((rx, ry))
        }
    }
}

// ---------------------------------------------------------------------------
// Distance utilities
// ---------------------------------------------------------------------------

/// Euclidean distance between two row-major vectors of length `dim`.
#[inline]
fn euclidean_dist(a: &[f64], b: &[f64], dim: usize) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..dim {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

/// Compute the full pairwise distance matrix (n×n, row-major) for `data` of shape (n, dim).
fn pairwise_distances(data: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_dist(
                &data[i * dim..(i + 1) * dim],
                &data[j * dim..(j + 1) * dim],
                dim,
            );
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// H0 via union-find (connected components)
// ---------------------------------------------------------------------------

/// Compute H0 persistence pairs using union-find on the sorted edge list.
///
/// Each point starts as its own component (born at ε = 0). When an edge merges
/// two components, the younger component dies at the edge's filtration value.
///
/// Returns `(h0_pairs, betti_0)` where `betti_0` is the number of essential
/// (infinite-lived) H0 classes.
fn compute_h0(n: usize, dist_matrix: &[f64], max_radius: f64) -> (Vec<PersistencePair>, usize) {
    // Collect all edges (i,j) with i < j sorted by distance ascending.
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist_matrix[i * n + j];
            edges.push((d, i, j));
        }
    }
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut uf = UnionFind::new(n);
    let mut pairs: Vec<PersistencePair> = Vec::new();

    for (d, i, j) in &edges {
        // Only process edges within max_radius.
        if *d > max_radius {
            break;
        }
        if let Some((killed, _surviving)) = uf.union(*i, *j) {
            // The component that dies is the one with higher birth (younger),
            // but since all births are 0 for H0 from points, any merge gives (0, d).
            let birth = uf.birth[killed];
            pairs.push(PersistencePair {
                birth,
                death: *d,
                dimension: 0,
            });
        }
    }

    // Count surviving components = betti_0.
    // After the union-find, roots represent surviving components.
    let mut roots_seen = std::collections::HashSet::new();
    for i in 0..n {
        roots_seen.insert(uf.find(i));
    }
    let betti_0 = roots_seen.len();

    // Each surviving component gets an essential pair (0, ∞).
    for _ in 0..betti_0 {
        pairs.push(PersistencePair {
            birth: 0.0,
            death: f64::INFINITY,
            dimension: 0,
        });
    }

    (pairs, betti_0)
}

// ---------------------------------------------------------------------------
// H1 via boundary matrix reduction
// ---------------------------------------------------------------------------

/// A 1-simplex (edge) in the Vietoris-Rips filtration.
#[derive(Debug, Clone, Copy)]
struct Edge {
    i: usize,
    j: usize,
    filtration: f64,
}

/// A 2-simplex (triangle) in the Vietoris-Rips filtration.
#[derive(Debug, Clone)]
struct Triangle {
    /// Boundary edges as sorted edge indices [e0, e1, e2] with e0 < e1 < e2.
    boundary: [usize; 3],
    filtration: f64,
}

/// Compute H1 persistence pairs via standard boundary matrix column reduction.
///
/// The algorithm:
/// 1. Sort edges by filtration value → index maps.
/// 2. For each triangle (2-simplex), build boundary as sorted edge indices.
/// 3. Represent each column as a sorted set of row indices (sparse).
/// 4. Column reduce: for each triangle column (left to right), if pivot matches
///    a previously reduced column, XOR (subtract mod 2) and continue.
/// 5. After reduction: paired edges give (birth, death) = (edge filtration, triangle filtration).
///    Unpaired edges that don't appear as pivots → essential H1 (birth, ∞).
fn compute_h1(n: usize, dist_matrix: &[f64], max_radius: f64) -> (Vec<PersistencePair>, usize) {
    // --- Build edge list sorted by filtration value ---
    let mut raw_edges: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist_matrix[i * n + j];
            if d <= max_radius {
                raw_edges.push((d, i, j));
            }
        }
    }
    raw_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Map (i,j) → edge index.
    let mut edge_index_map: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    let mut edges: Vec<Edge> = Vec::with_capacity(raw_edges.len());
    for (idx, (d, i, j)) in raw_edges.iter().enumerate() {
        edge_index_map.insert((*i, *j), idx);
        edges.push(Edge {
            i: *i,
            j: *j,
            filtration: *d,
        });
    }

    let n_edges = edges.len();
    if n_edges == 0 {
        return (Vec::new(), 0);
    }

    // --- Build triangle list sorted by filtration value ---
    let mut triangles: Vec<Triangle> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d_ij = dist_matrix[i * n + j];
            if d_ij > max_radius {
                continue;
            }
            for k in (j + 1)..n {
                let d_ik = dist_matrix[i * n + k];
                let d_jk = dist_matrix[j * n + k];
                if d_ik > max_radius || d_jk > max_radius {
                    continue;
                }
                // All three edges exist within max_radius → valid 2-simplex.
                let filt = d_ij.max(d_ik).max(d_jk);

                // Get edge indices for the three boundary edges.
                // Boundary of triangle (i,j,k) = [ij, ik, jk] with i<j<k.
                let e_ij = edge_index_map[&(i, j)];
                let e_ik = edge_index_map[&(i, k)];
                let e_jk = edge_index_map[&(j, k)];

                // Sorted ascending for canonical representation.
                let mut b = [e_ij, e_ik, e_jk];
                b.sort_unstable();

                triangles.push(Triangle {
                    boundary: b,
                    filtration: filt,
                });
            }
        }
    }

    // Sort triangles by filtration value.
    triangles.sort_by(|a, b| {
        a.filtration
            .partial_cmp(&b.filtration)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_triangles = triangles.len();
    if n_triangles == 0 {
        // No triangles → no H1 deaths, all cycle-creating edges are essential.
        // But without triangles there are no H1 births either in standard persistence.
        return (Vec::new(), 0);
    }

    // --- Boundary matrix column reduction (sparse, F_2 coefficients) ---
    //
    // Columns correspond to triangles (2-simplices).
    // Rows correspond to edges (1-simplices).
    // D[col] = sorted list of boundary edge indices (the non-zero rows).
    //
    // We work modulo 2: XOR is the field operation.
    // We only need the pivot (lowest row index) of each reduced column.

    // `columns[t]` = sorted set of edge indices forming the boundary of triangle t.
    let mut columns: Vec<Vec<usize>> = triangles.iter().map(|tri| tri.boundary.to_vec()).collect();

    // `pivot_col[row_idx]` → Some(col_idx) when a column with that pivot has been seen.
    let mut pivot_col: Vec<Option<usize>> = vec![None; n_edges];

    // `paired_edge[col]` = edge index that this column was paired with (pivot after reduction).
    // `None` means the column reduced to zero → no H1 pairing for this triangle.
    let mut paired_edge_for_col: Vec<Option<usize>> = vec![None; n_triangles];

    for col in 0..n_triangles {
        // Reduce column `col`: repeatedly XOR with the column that has the same pivot,
        // until no such column exists or the column becomes zero.
        loop {
            if columns[col].is_empty() {
                break;
            }
            let pivot = *columns[col].last().expect("non-empty");
            match pivot_col[pivot] {
                None => {
                    // No prior column has this pivot → register and stop.
                    pivot_col[pivot] = Some(col);
                    paired_edge_for_col[col] = Some(pivot);
                    break;
                }
                Some(prev_col) => {
                    // XOR columns[col] with columns[prev_col] (symmetric difference).
                    let other = columns[prev_col].clone();
                    sym_diff_inplace(&mut columns[col], &other);
                }
            }
        }
    }

    // --- Extract H1 pairs ---
    // A pair (edge_idx, triangle_idx) gives H1 birth = edges[edge_idx].filtration,
    //                                          H1 death = triangles[triangle_idx].filtration.
    //
    // The set of "killed" edge indices = pivots of non-zero reduced columns.
    let mut killed_edges: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut h1_pairs: Vec<PersistencePair> = Vec::new();

    for col in 0..n_triangles {
        if let Some(edge_idx) = paired_edge_for_col[col] {
            killed_edges.insert(edge_idx);
            let birth = edges[edge_idx].filtration;
            let death = triangles[col].filtration;
            // Include all pairs (birth ≤ death holds by construction since the triangle
            // enters the filtration no earlier than any of its edges).
            h1_pairs.push(PersistencePair {
                birth,
                death,
                dimension: 1,
            });
        }
    }

    // Essential H1 classes: edges not killed by any triangle.
    // An edge is a H1 generator if it is in the cycle space.
    // In the Vietoris-Rips complex, "generator of H1" edges = edges not used as pivot.
    // However, not every unpaired edge is a genuine H1 generator; we must check if the edge
    // is a loop (not killed and creates a cycle in H0).
    //
    // Actually, in standard persistence: the essential H1 classes correspond to
    // unpaired edges (those whose reduced column was never empty and whose row
    // index never appeared as pivot). These are edges not in `killed_edges` and
    // not a boundary edge of any 2-simplex that reduced to that pivot.
    //
    // For simplicity (and correctness for common inputs): an edge is an H1 generator
    // if and only if it is not in `killed_edges` AND it would create a cycle
    // (i.e., the union-find would reject it at the time it's added — both endpoints
    // already connected). We use a second union-find pass.
    let mut uf2 = UnionFind::new(n);
    // Re-process edges in sorted order; edges that create cycles are potential H1 generators.
    let mut cycle_edges: Vec<usize> = Vec::new();
    for (idx, edge) in edges.iter().enumerate() {
        if uf2.union(edge.i, edge.j).is_none() {
            // This edge created a cycle.
            if !killed_edges.contains(&idx) {
                // Not killed by any triangle → essential H1 class.
                cycle_edges.push(idx);
            }
        }
    }
    let betti_1 = cycle_edges.len();
    for e_idx in &cycle_edges {
        h1_pairs.push(PersistencePair {
            birth: edges[*e_idx].filtration,
            death: f64::INFINITY,
            dimension: 1,
        });
    }

    (h1_pairs, betti_1)
}

/// Compute the symmetric difference (XOR) of two sorted vectors of `usize`, in-place.
///
/// Elements appearing in exactly one of the inputs are retained; elements in both are removed.
fn sym_diff_inplace(a: &mut Vec<usize>, b: &[usize]) {
    // Merge-based O(|a| + |b|) symmetric difference.
    let mut result: Vec<usize> = Vec::with_capacity(a.len() + b.len());
    let mut ai = 0;
    let mut bi = 0;
    while ai < a.len() && bi < b.len() {
        match a[ai].cmp(&b[bi]) {
            std::cmp::Ordering::Less => {
                result.push(a[ai]);
                ai += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(b[bi]);
                bi += 1;
            }
            std::cmp::Ordering::Equal => {
                // In both → cancel (F_2 arithmetic).
                ai += 1;
                bi += 1;
            }
        }
    }
    result.extend_from_slice(&a[ai..]);
    result.extend_from_slice(&b[bi..]);
    *a = result;
}

// ---------------------------------------------------------------------------
// Public API: Vietoris-Rips Persistent Homology
// ---------------------------------------------------------------------------

/// Compute the Vietoris-Rips persistent homology of a point cloud.
///
/// # Parameters
///
/// - `data`     — row-major float array of shape `(n_points, dim)`.
/// - `n_points` — number of points in `data`.
/// - `dim`      — embedding dimension (number of features per point).
/// - `config`   — filtration configuration.
///
/// # Returns
///
/// A [`PersistenceDiagram`] containing H0 and (optionally) H1 birth-death pairs,
/// Betti numbers, and the filtration radius used.
///
/// # Errors
///
/// Returns [`ManifoldError::EmptyInput`] if `n_points == 0` or `dim == 0`.
/// Returns [`ManifoldError::ShapeMismatch`] if `data.len() != n_points * dim`.
/// Returns [`ManifoldError::InvalidParameter`] for invalid config values.
pub fn vietoris_rips_persistence(
    data: &[f64],
    n_points: usize,
    dim: usize,
    config: &VietorisRipsConfig,
) -> ManifoldResult<PersistenceDiagram> {
    // --- Validation ---
    if n_points == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if data.len() != n_points * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_points, dim],
            got: vec![data.len()],
        });
    }
    if config.max_dimension > 1 {
        return Err(ManifoldError::InvalidParameter {
            name: "max_dimension".to_string(),
            reason: "only H0 (0) and H1 (1) are supported".to_string(),
        });
    }

    // Trivial case: single point.
    if n_points == 1 {
        let pairs = vec![PersistencePair {
            birth: 0.0,
            death: f64::INFINITY,
            dimension: 0,
        }];
        return Ok(PersistenceDiagram {
            pairs,
            betti_0: 1,
            betti_1: 0,
            max_filtration: 0.0,
        });
    }

    // --- Pairwise distances ---
    let dist_matrix = pairwise_distances(data, n_points, dim);

    // --- Auto-select max_radius ---
    let diameter = dist_matrix.iter().cloned().fold(0.0_f64, f64::max);
    let max_radius = if config.max_radius > 0.0 {
        config.max_radius
    } else {
        diameter / 4.0
    };

    // --- H0 computation ---
    let (mut pairs, betti_0) = compute_h0(n_points, &dist_matrix, max_radius);

    // --- H1 computation ---
    let betti_1 = if config.max_dimension >= 1 {
        let (h1_pairs, b1) = compute_h1(n_points, &dist_matrix, max_radius);
        pairs.extend(h1_pairs);
        b1
    } else {
        0
    };

    // Sort all pairs: by dimension first, then by birth.
    pairs.sort_by(|a, b| {
        a.dimension.cmp(&b.dimension).then(
            a.birth
                .partial_cmp(&b.birth)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });

    Ok(PersistenceDiagram {
        pairs,
        betti_0,
        betti_1,
        max_filtration: max_radius,
    })
}

// ---------------------------------------------------------------------------
// Public API: Betti numbers at a given filtration value
// ---------------------------------------------------------------------------

/// Compute Betti numbers (β₀, β₁) at a specific filtration value.
///
/// A feature is "alive" at filtration value `ε` if `birth ≤ ε < death`.
///
/// # Returns
///
/// `(betti_0, betti_1)` — number of living H0 and H1 classes at `at_filtration`.
#[must_use]
pub fn persistence_betti(diagram: &PersistenceDiagram, at_filtration: f64) -> (usize, usize) {
    let mut b0 = 0usize;
    let mut b1 = 0usize;
    for pair in &diagram.pairs {
        let alive =
            pair.birth <= at_filtration && (pair.death.is_infinite() || at_filtration < pair.death);
        if alive {
            match pair.dimension {
                0 => b0 += 1,
                1 => b1 += 1,
                _ => {}
            }
        }
    }
    (b0, b1)
}

// ---------------------------------------------------------------------------
// Public API: Bottleneck distance between persistence diagrams
// ---------------------------------------------------------------------------

/// Compute the bottleneck distance between two persistence diagrams in a given dimension.
///
/// The bottleneck distance is the infimum over all matchings between the two diagrams
/// (including matching to the diagonal) of the maximum cost:
///   W_∞(D₁, D₂) = inf_{γ} sup_{x ∈ D₁} ‖x − γ(x)‖_∞
///
/// Points may be matched to the diagonal (their nearest point on the diagonal is
/// ((b+d)/2, (b+d)/2)); the cost of this match is half the persistence `(d−b)/2`.
///
/// # Algorithm
///
/// We use the O(n² log n) approximation via binary search + bipartite matching.
/// For correctness and simplicity (given modest diagram sizes in TDA applications),
/// we implement the exact quadratic algorithm: try all candidate bottleneck distances
/// from the set of pairwise L∞ distances between points (and point-diagonal distances),
/// and binary-search for the minimum threshold δ that allows a perfect matching.
/// Matching feasibility is checked with a greedy augmenting-path (Hopcroft-Karp style).
///
/// For empty diagrams, the distance equals the maximum persistence/2 of the other diagram.
#[must_use]
pub fn bottleneck_distance(
    diag1: &PersistenceDiagram,
    diag2: &PersistenceDiagram,
    dimension: usize,
) -> f64 {
    let pts1: Vec<(f64, f64)> = diag1
        .pairs
        .iter()
        .filter(|p| p.dimension == dimension && !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();

    let pts2: Vec<(f64, f64)> = diag2
        .pairs
        .iter()
        .filter(|p| p.dimension == dimension && !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();

    bottleneck_distance_pts(&pts1, &pts2)
}

/// L∞ distance between two diagram points, or between a point and the diagonal.
#[inline]
fn l_inf_point_to_point(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).abs().max((a.1 - b.1).abs())
}

#[inline]
fn l_inf_point_to_diagonal(p: (f64, f64)) -> f64 {
    (p.1 - p.0) / 2.0
}

/// Check if a perfect matching of cost ≤ `threshold` exists between `pts1` and `pts2`.
///
/// Points not matched to each other are matched to the diagonal; the cost is
/// `(d − b) / 2`. We require *all* points (on both sides) to be matched within threshold.
///
/// Uses greedy augmenting-path matching (DFS-based).
fn matching_feasible(pts1: &[(f64, f64)], pts2: &[(f64, f64)], threshold: f64) -> bool {
    // First check: all points on each side must be matchable to their diagonal
    // OR to some point on the other side within the threshold.
    for &p in pts1 {
        if l_inf_point_to_diagonal(p) > threshold {
            // Must be matched to some point in pts2.
            if !pts2
                .iter()
                .any(|&q| l_inf_point_to_point(p, q) <= threshold)
            {
                return false;
            }
        }
    }
    for &q in pts2 {
        if l_inf_point_to_diagonal(q) > threshold
            && !pts1
                .iter()
                .any(|&p| l_inf_point_to_point(p, q) <= threshold)
        {
            return false;
        }
    }

    // Build adjacency: for pts1[i] and pts2[j], edge exists if cost ≤ threshold.
    // Also include "diagonal" option for each point (as a virtual node).
    let n2 = pts2.len();

    // Each pts1[i] can be matched to pts2[j] or the diagonal.
    // Each pts2[j] can be matched to pts1[i] or the diagonal.
    // We need a complete matching of all "hard" points.

    // Determine which points must be matched to real counterparts (not diagonal).
    let must_match_1: Vec<bool> = pts1
        .iter()
        .map(|&p| l_inf_point_to_diagonal(p) > threshold)
        .collect();
    let must_match_2: Vec<bool> = pts2
        .iter()
        .map(|&q| l_inf_point_to_diagonal(q) > threshold)
        .collect();

    // Use augmenting-path matching for the bipartite graph of hard points.
    // Match indices in pts1 that are "must_match" to indices in pts2.
    let n1 = pts1.len();
    let hard1: Vec<usize> = (0..n1).filter(|&i| must_match_1[i]).collect();
    let hard2: Vec<usize> = (0..n2).filter(|&j| must_match_2[j]).collect();

    // Build adjacency for hard1 → hard2 and hard1 → pts2 within threshold.
    // Actually we need to match ALL hard1 nodes to some pts2 node within threshold,
    // and ALL hard2 nodes to some pts1 node (already checked above).
    // Additionally, soft pts1 nodes can "absorb" hard pts2 nodes as needed.
    //
    // Simplified formulation: augmented bipartite matching where left = pts1, right = pts2 ∪ {diagonal}.
    // Each pts1[i] → pts2[j] if cost ≤ threshold; each pts1[i] → diagonal always (if diagonal_cost ≤ threshold).

    // Use Hungarian / augmenting-path matching for the critical subset.
    let n_hard1 = hard1.len();
    if n_hard1 == 0 {
        // All pts1 can go to diagonal; check all hard2 can be matched.
        return hard2.is_empty()
            || hard2.iter().all(|&j| {
                pts1.iter()
                    .any(|&p| l_inf_point_to_point(p, pts2[j]) <= threshold)
            });
    }

    // `match_right[j]` = index in `hard1` currently matched to pts2[j], or usize::MAX.
    let mut match_right: Vec<usize> = vec![usize::MAX; n2];

    let mut matched_count = 0;
    for &i1 in &hard1 {
        let mut visited = vec![false; n2];
        if augment(i1, pts1, pts2, threshold, &mut match_right, &mut visited) {
            matched_count += 1;
        }
    }

    // All hard1 nodes must be matched.
    if matched_count < n_hard1 {
        return false;
    }

    // Now ensure all hard2 nodes are matched (either as match_right target or to diagonal via pts1).
    for &j2 in &hard2 {
        if match_right[j2] == usize::MAX {
            // This hard2 node is not matched to any pts1 node.
            return false;
        }
    }

    true
}

/// DFS augmenting path from pts1[`i1`] to any unmatched pts2[j] reachable within `threshold`.
fn augment(
    i1: usize,
    pts1: &[(f64, f64)],
    pts2: &[(f64, f64)],
    threshold: f64,
    match_right: &mut Vec<usize>,
    visited: &mut Vec<bool>,
) -> bool {
    for j2 in 0..pts2.len() {
        if visited[j2] {
            continue;
        }
        if l_inf_point_to_point(pts1[i1], pts2[j2]) > threshold {
            continue;
        }
        visited[j2] = true;
        // Try to match i1 → j2: either j2 is free, or we can re-route its current match.
        let prev = match_right[j2];
        if prev == usize::MAX || augment(prev, pts1, pts2, threshold, match_right, visited) {
            match_right[j2] = i1;
            return true;
        }
    }
    false
}

/// Core bottleneck distance computation for finite diagram points.
fn bottleneck_distance_pts(pts1: &[(f64, f64)], pts2: &[(f64, f64)]) -> f64 {
    if pts1.is_empty() && pts2.is_empty() {
        return 0.0;
    }

    // Collect all candidate threshold values:
    // - pairwise L∞ distances between pts1 and pts2,
    // - diagonal distances for all points in both diagrams.
    let mut candidates: Vec<f64> = Vec::new();
    for &p in pts1 {
        candidates.push(l_inf_point_to_diagonal(p));
        for &q in pts2 {
            candidates.push(l_inf_point_to_point(p, q));
        }
    }
    for &q in pts2 {
        candidates.push(l_inf_point_to_diagonal(q));
    }

    // Add small perturbations to handle floating-point boundary issues.
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

    if candidates.is_empty() {
        return 0.0;
    }

    // Binary search: find minimum δ such that matching is feasible.
    let mut lo = 0usize;
    let mut hi = candidates.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if matching_feasible(pts1, pts2, candidates[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    if lo < candidates.len() {
        candidates[lo]
    } else {
        // Fallback: max diagonal distance (should not happen for valid inputs).
        candidates.iter().cloned().fold(0.0_f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// Mapper algorithm
// ---------------------------------------------------------------------------

/// Compute the centroid of a point cloud.
fn compute_centroid(data: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut centroid = vec![0.0_f64; dim];
    for i in 0..n {
        for d in 0..dim {
            centroid[d] += data[i * dim + d];
        }
    }
    let n_f = n as f64;
    for c in centroid.iter_mut() {
        *c /= n_f;
    }
    centroid
}

/// Filter function: projection of each point on the highest-variance coordinate axis.
///
/// This is a fast approximation to the first principal component: we find the coordinate
/// dimension with the largest variance and use the projected value on that axis as the
/// filter. When all dimensions have equal variance (e.g. a circle), we fall back to
/// the distance-to-centroid function to avoid a degenerate (flat) filter.
fn compute_filter_values(data: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let centroid = compute_centroid(data, n, dim);

    // Compute per-dimension variance.
    let mut variances = vec![0.0_f64; dim];
    for i in 0..n {
        for d in 0..dim {
            let diff = data[i * dim + d] - centroid[d];
            variances[d] += diff * diff;
        }
    }

    // Pick the dimension with the highest variance.
    let best_dim = variances
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(d, _)| d)
        .unwrap_or(0);

    let max_var = variances[best_dim];
    let f_range_sq = max_var / n as f64;

    if f_range_sq < 1e-12 {
        // Degenerate: all points nearly on same projection → fall back to centroid distance.
        return (0..n)
            .map(|i| euclidean_dist(&data[i * dim..(i + 1) * dim], &centroid, dim))
            .collect();
    }

    // Project all points on the best axis.
    (0..n).map(|i| data[i * dim + best_dim]).collect()
}

/// k-means++ initialisation: select `k` seeds with probability ∝ squared distance to nearest seed.
fn kmeans_plusplus_seeds(
    points: &[usize],
    data: &[f64],
    dim: usize,
    k: usize,
    rng: &mut LcgRng,
) -> Vec<Vec<f64>> {
    let m = points.len();
    debug_assert!(k <= m);

    let mut seeds: Vec<Vec<f64>> = Vec::with_capacity(k);

    // First seed: uniformly random.
    let first_idx = rng.next_usize(m);
    let first_pt = points[first_idx];
    seeds.push(data[first_pt * dim..(first_pt + 1) * dim].to_vec());

    for _ in 1..k {
        // Compute squared distance from each point to its nearest seed.
        let mut dists: Vec<f64> = points
            .iter()
            .map(|&pt| {
                seeds
                    .iter()
                    .map(|s| {
                        let mut sq = 0.0_f64;
                        for d in 0..dim {
                            let diff = data[pt * dim + d] - s[d];
                            sq += diff * diff;
                        }
                        sq
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        // Sample proportional to squared distance.
        let total: f64 = dists.iter().sum();
        if total <= 0.0 {
            // All points are at seeds; just pick the next one.
            let idx = rng.next_usize(m);
            let pt = points[idx];
            seeds.push(data[pt * dim..(pt + 1) * dim].to_vec());
        } else {
            let mut r = rng.next_f64() * total;
            let mut chosen = m - 1;
            for (ii, d) in dists.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    chosen = ii;
                    break;
                }
            }
            let pt = points[chosen];
            seeds.push(data[pt * dim..(pt + 1) * dim].to_vec());
        }
        // Suppress unused-assignment warning.
        let _ = dists.iter_mut();
    }

    seeds
}

/// Run k-means on a subset of points (given by `point_indices`) for at most `max_iter` iterations.
///
/// Returns cluster label for each element of `point_indices`.
fn kmeans_on_subset(
    point_indices: &[usize],
    data: &[f64],
    dim: usize,
    k: usize,
    max_iter: usize,
    rng: &mut LcgRng,
) -> Vec<usize> {
    let m = point_indices.len();
    if m == 0 {
        return Vec::new();
    }
    // If k ≥ m, each point is its own cluster.
    let actual_k = k.min(m);

    // k-means++ seeds.
    let mut centers = kmeans_plusplus_seeds(point_indices, data, dim, actual_k, rng);

    let mut labels = vec![0usize; m];
    let mut changed = true;
    let mut iter = 0;

    while changed && iter < max_iter {
        changed = false;
        iter += 1;

        // Assignment step.
        for (ii, &pt) in point_indices.iter().enumerate() {
            let mut best_c = 0;
            let mut best_d = f64::INFINITY;
            for (c, center) in centers.iter().enumerate() {
                let mut sq = 0.0_f64;
                for d in 0..dim {
                    let diff = data[pt * dim + d] - center[d];
                    sq += diff * diff;
                }
                if sq < best_d {
                    best_d = sq;
                    best_c = c;
                }
            }
            if labels[ii] != best_c {
                labels[ii] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step: recompute centers.
        let mut new_centers = vec![vec![0.0_f64; dim]; actual_k];
        let mut counts = vec![0usize; actual_k];
        for (ii, &pt) in point_indices.iter().enumerate() {
            let c = labels[ii];
            for d in 0..dim {
                new_centers[c][d] += data[pt * dim + d];
            }
            counts[c] += 1;
        }
        for c in 0..actual_k {
            if counts[c] > 0 {
                let n_f = counts[c] as f64;
                for val in new_centers[c].iter_mut() {
                    *val /= n_f;
                }
            } else {
                // Empty cluster: reinitialise from a random point.
                let idx = rng.next_usize(m);
                let pt = point_indices[idx];
                new_centers[c] = data[pt * dim..(pt + 1) * dim].to_vec();
                changed = true;
            }
        }
        centers = new_centers;
    }

    labels
}

/// Build the Mapper topological graph from a point cloud.
///
/// # Parameters
///
/// - `data`     — row-major float array of shape `(n_points, dim)`.
/// - `n_points` — number of points.
/// - `dim`      — number of features per point.
/// - `config`   — Mapper algorithm configuration.
///
/// # Returns
///
/// A [`MapperGraph`] with nodes (one per interval-cluster pair) and edges between nodes
/// that share at least one data point.
///
/// # Errors
///
/// Returns [`ManifoldError::EmptyInput`] if `n_points == 0` or `dim == 0`.
/// Returns [`ManifoldError::InvalidParameter`] for invalid configuration values.
pub fn mapper(
    data: &[f64],
    n_points: usize,
    dim: usize,
    config: &MapperConfig,
) -> ManifoldResult<MapperGraph> {
    // --- Validation ---
    if n_points == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if data.len() != n_points * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n_points, dim],
            got: vec![data.len()],
        });
    }
    if config.n_intervals == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_intervals".to_string(),
            reason: "must be at least 1".to_string(),
        });
    }
    if config.n_clusters == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_clusters".to_string(),
            reason: "must be at least 1".to_string(),
        });
    }
    if config.overlap <= 0.0 || config.overlap >= 1.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "overlap".to_string(),
            reason: "must be in (0, 1) exclusive".to_string(),
        });
    }

    // --- Filter function: highest-variance coordinate projection (with centroid-distance fallback) ---
    let filter_vals = compute_filter_values(data, n_points, dim);

    let f_min = filter_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_max = filter_vals
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // If all points coincide (f_min == f_max), put everything in one cluster.
    let f_range = f_max - f_min;

    // --- Cover: n_intervals overlapping intervals ---
    //
    // Interval width w = f_range / (n_intervals - overlap * (n_intervals - 1))
    // Step between interval starts s = w * (1 - overlap)
    let n = config.n_intervals;
    let ovlp = config.overlap;

    // Compute interval boundaries.
    let intervals: Vec<(f64, f64)> = if f_range < 1e-12 || n == 1 {
        // Degenerate: single interval covering everything.
        vec![(f_min - 1e-9, f_max + 1e-9)]
    } else {
        let step = f_range / (n as f64 - ovlp * (n as f64 - 1.0));
        let width = step / (1.0 - ovlp);
        (0..n)
            .map(|k| {
                let lo = f_min + k as f64 * step;
                let hi = lo + width;
                (lo, hi)
            })
            .collect()
    };

    let mut rng = LcgRng::new(config.seed);

    // --- Cluster within each interval ---
    let mut nodes: Vec<MapperNode> = Vec::new();

    for (interval_idx, &(lo, hi)) in intervals.iter().enumerate() {
        // Collect points whose filter value falls in [lo, hi].
        let pts_in_interval: Vec<usize> = (0..n_points)
            .filter(|&i| filter_vals[i] >= lo && filter_vals[i] <= hi)
            .collect();

        if pts_in_interval.is_empty() {
            continue;
        }

        let k = config.n_clusters.min(pts_in_interval.len());
        let labels = kmeans_on_subset(&pts_in_interval, data, dim, k, 100, &mut rng);

        // Group points by cluster label.
        let mut cluster_points: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (local_idx, &global_idx) in pts_in_interval.iter().enumerate() {
            let c = labels[local_idx];
            cluster_points[c].push(global_idx);
        }

        for (cluster_idx, pts) in cluster_points.into_iter().enumerate() {
            if pts.is_empty() {
                continue;
            }
            let size = pts.len();
            nodes.push(MapperNode {
                interval_idx,
                cluster_idx,
                point_indices: pts,
                size,
            });
        }
    }

    // --- Build edges: connect nodes from adjacent intervals that share points ---
    let n_nodes = nodes.len();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut seen_edges: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();

    for u in 0..n_nodes {
        for v in (u + 1)..n_nodes {
            // Only connect nodes from adjacent intervals (or the same interval with overlap).
            let iu = nodes[u].interval_idx;
            let iv = nodes[v].interval_idx;
            let interval_gap = iu.abs_diff(iv);
            if interval_gap > 1 {
                continue;
            }
            // Check if they share at least one point.
            let set_u: std::collections::HashSet<usize> =
                nodes[u].point_indices.iter().cloned().collect();
            let shares = nodes[v].point_indices.iter().any(|p| set_u.contains(p));
            if shares {
                let key = (u.min(v), u.max(v));
                if seen_edges.insert(key) {
                    edges.push(key);
                }
            }
        }
    }

    Ok(MapperGraph { nodes, edges })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build an equilateral triangle with side length 1.
    fn triangle_data() -> (Vec<f64>, usize, usize) {
        // Three vertices of an equilateral triangle (side = 1.0).
        let data = vec![
            0.0_f64,
            0.0, // vertex 0
            1.0,
            0.0, // vertex 1
            0.5,
            0.866_025_403_784_438_6, // vertex 2
        ];
        (data, 3, 2)
    }

    // Helper: 4 isolated points far apart.
    fn four_isolated_points() -> (Vec<f64>, usize, usize) {
        let data = vec![
            0.0_f64, 0.0, // point 0
            10.0, 0.0, // point 1
            0.0, 10.0, // point 2
            10.0, 10.0, // point 3
        ];
        (data, 4, 2)
    }

    // -----------------------------------------------------------------------
    // Test 1: Single point — trivial persistence
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_point_trivial() {
        // Arbitrary 2-D point coordinates for a single-point fixture.
        let data = vec![3.5_f64, 2.72];
        let config = VietorisRipsConfig {
            max_radius: 10.0,
            max_dimension: 1,
            n_points: 1,
        };
        let diag = vietoris_rips_persistence(&data, 1, 2, &config).expect("single point");
        assert_eq!(diag.betti_0, 1, "single point: betti_0 = 1");
        assert_eq!(diag.betti_1, 0, "single point: betti_1 = 0");
        assert_eq!(diag.pairs.len(), 1, "single point: exactly one pair");
        assert!(
            diag.pairs[0].is_essential(),
            "single point pair is essential"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: Four isolated points at ε=0 — betti_0=4, betti_1=0
    // -----------------------------------------------------------------------
    #[test]
    fn test_four_isolated_points_small_radius() {
        let (data, n, dim) = four_isolated_points();
        let config = VietorisRipsConfig {
            max_radius: 0.5, // No edges appear (all distances ≥ 10)
            max_dimension: 1,
            n_points: n,
        };
        let diag = vietoris_rips_persistence(&data, n, dim, &config).expect("four isolated");
        assert_eq!(diag.betti_0, 4, "four isolated: betti_0 = 4");
        assert_eq!(diag.betti_1, 0, "four isolated: betti_1 = 0");
    }

    // -----------------------------------------------------------------------
    // Test 3: persistence_betti at ε=0 — all n_points components, 0 loops
    // -----------------------------------------------------------------------
    #[test]
    fn test_persistence_betti_at_zero() {
        let (data, n, dim) = four_isolated_points();
        let config = VietorisRipsConfig {
            max_radius: 100.0,
            max_dimension: 1,
            n_points: n,
        };
        let diag = vietoris_rips_persistence(&data, n, dim, &config).expect("betti at 0");
        let (b0, b1) = persistence_betti(&diag, 0.0);
        assert_eq!(b0, n, "at ε=0 each point is its own component");
        assert_eq!(b1, 0, "at ε=0 no loops");
    }

    // -----------------------------------------------------------------------
    // Test 4: Triangle — H0 has 2 finite pairs + 1 essential, H1 has 1 finite pair
    // -----------------------------------------------------------------------
    #[test]
    fn test_triangle_h0_h1() {
        let (data, n, dim) = triangle_data();
        let config = VietorisRipsConfig {
            max_radius: 2.0,
            max_dimension: 1,
            n_points: n,
        };
        let diag = vietoris_rips_persistence(&data, n, dim, &config).expect("triangle");
        // H0: 3 points start, 2 merges → 2 finite pairs + 1 essential
        let h0_pairs = diag.pairs_in_dimension(0);
        let h0_finite = h0_pairs.iter().filter(|p| !p.is_essential()).count();
        let h0_essential = h0_pairs.iter().filter(|p| p.is_essential()).count();
        assert_eq!(h0_finite, 2, "triangle: 2 finite H0 pairs");
        assert_eq!(h0_essential, 1, "triangle: 1 essential H0 pair");
        // H1: triangle kills 1 loop → at least 1 finite H1 pair
        let h1_pairs = diag.pairs_in_dimension(1);
        let h1_finite = h1_pairs.iter().filter(|p| !p.is_essential()).count();
        assert!(
            h1_finite >= 1,
            "triangle: ≥1 finite H1 pair (loop killed by triangle)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: All points connected at large ε — betti_0=1
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_connected_large_radius() {
        let (data, n, dim) = four_isolated_points();
        let config = VietorisRipsConfig {
            max_radius: 100.0,
            max_dimension: 0,
            n_points: n,
        };
        let diag = vietoris_rips_persistence(&data, n, dim, &config).expect("all connected");
        assert_eq!(diag.betti_0, 1, "all connected: betti_0 = 1");
    }

    // -----------------------------------------------------------------------
    // Test 6: Two clusters — betti_0=2 at medium ε, betti_0=1 at large ε
    // -----------------------------------------------------------------------
    #[test]
    fn test_two_clusters_betti() {
        // Two tight clusters separated by large gap.
        let data = vec![
            0.0_f64, 0.0, // cluster A
            0.1, 0.0, 0.0, 0.1, 5.0, 5.0, // cluster B
            5.1, 5.0, 5.0, 5.1,
        ];
        let n = 6;
        let dim = 2;
        let config = VietorisRipsConfig {
            max_radius: 20.0,
            max_dimension: 0,
            n_points: n,
        };
        let diag = vietoris_rips_persistence(&data, n, dim, &config).expect("two clusters");
        // At ε = 0.15: intra-cluster edges exist (~0.1), inter-cluster don't (~7).
        let (b0_medium, _) = persistence_betti(&diag, 0.15);
        assert_eq!(b0_medium, 2, "two clusters: betti_0=2 at medium ε");
        // At ε = 20: all connected.
        let (b0_large, _) = persistence_betti(&diag, 20.0);
        assert_eq!(b0_large, 1, "two clusters: betti_0=1 at large ε");
    }

    // -----------------------------------------------------------------------
    // Test 7: bottleneck_distance(diag, diag) = 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_bottleneck_distance_self() {
        let (data, n, dim) = triangle_data();
        let config = VietorisRipsConfig {
            max_radius: 2.0,
            max_dimension: 1,
            n_points: n,
        };
        let diag = vietoris_rips_persistence(&data, n, dim, &config).expect("triangle");
        let d0 = bottleneck_distance(&diag, &diag, 0);
        let d1 = bottleneck_distance(&diag, &diag, 1);
        assert!(d0.abs() < 1e-10, "bottleneck(diag, diag, 0) = 0, got {d0}");
        assert!(d1.abs() < 1e-10, "bottleneck(diag, diag, 1) = 0, got {d1}");
    }

    // -----------------------------------------------------------------------
    // Test 8: bottleneck_distance(empty, nonempty) > 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_bottleneck_distance_empty_vs_nonempty() {
        let (data, n, dim) = triangle_data();
        let config = VietorisRipsConfig {
            max_radius: 2.0,
            max_dimension: 1,
            n_points: n,
        };
        let diag_nonempty =
            vietoris_rips_persistence(&data, n, dim, &config).expect("nonempty triangle diagram");
        let diag_empty = PersistenceDiagram {
            pairs: Vec::new(),
            betti_0: 0,
            betti_1: 0,
            max_filtration: 2.0,
        };
        let d = bottleneck_distance(&diag_empty, &diag_nonempty, 0);
        assert!(d > 0.0, "bottleneck(empty, nonempty) > 0, got {d}");
    }

    // -----------------------------------------------------------------------
    // Test 9: PersistencePair birth ≤ death always
    // -----------------------------------------------------------------------
    #[test]
    fn test_birth_leq_death() {
        let data = vec![0.0_f64, 0.0, 1.0, 0.0, 0.5, 0.8, 2.0, 1.0, -1.0, 0.5];
        let n = 5;
        let dim = 2;
        let config = VietorisRipsConfig {
            max_radius: 5.0,
            max_dimension: 1,
            n_points: n,
        };
        let diag = vietoris_rips_persistence(&data, n, dim, &config).expect("5-point cloud");
        for pair in &diag.pairs {
            assert!(
                pair.birth <= pair.death,
                "birth ({}) > death ({}) in pair {:?}",
                pair.birth,
                pair.death,
                pair
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 10: max_radius=0 — all components isolated
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_radius_zero() {
        let data = vec![0.0_f64, 0.0, 1.0, 0.0, 2.0, 0.0];
        let n = 3;
        let dim = 2;
        let config = VietorisRipsConfig {
            max_radius: 1e-12, // effectively zero (edges at distance > 0 not included)
            max_dimension: 1,
            n_points: n,
        };
        let diag = vietoris_rips_persistence(&data, n, dim, &config).expect("radius zero");
        assert_eq!(
            diag.betti_0, n,
            "radius≈0: all isolated, betti_0 = n_points"
        );
        assert_eq!(diag.betti_1, 0, "radius≈0: no loops");
    }

    // -----------------------------------------------------------------------
    // Test 11: Mapper on ring data — cycle structure
    // -----------------------------------------------------------------------
    #[test]
    fn test_mapper_ring_has_cycle() {
        // 20 points evenly distributed on a circle of radius 1.
        let n = 20;
        let data: Vec<f64> = (0..n)
            .flat_map(|i| {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                [angle.cos(), angle.sin()]
            })
            .collect();

        let config = MapperConfig {
            n_intervals: 6,
            overlap: 0.5,
            n_clusters: 2,
            seed: 42,
        };
        let graph = mapper(&data, n, 2, &config).expect("ring mapper");

        // The graph should have nodes.
        assert!(!graph.nodes.is_empty(), "ring mapper: graph has nodes");

        // The total count of point assignments should be ≥ n (overlap means repetition).
        let total_pts: usize = graph.nodes.iter().map(|nd| nd.size).sum();
        assert!(
            total_pts >= n,
            "mapper node sizes sum = {total_pts} should be ≥ n = {n} due to overlap"
        );

        // The graph should have edges (ring topology).
        assert!(!graph.edges.is_empty(), "ring mapper: graph has edges");
    }

    // -----------------------------------------------------------------------
    // Test 12: Mapper node sizes sum ≥ n_points
    // -----------------------------------------------------------------------
    #[test]
    fn test_mapper_point_coverage() {
        let (data, n, dim) = four_isolated_points();
        let config = MapperConfig {
            n_intervals: 3,
            overlap: 0.3,
            n_clusters: 1,
            seed: 7,
        };
        let graph = mapper(&data, n, dim, &config).expect("coverage mapper");
        let total: usize = graph.nodes.iter().map(|nd| nd.size).sum();
        assert!(total >= n, "mapper coverage: total = {total}, n = {n}");
    }

    // -----------------------------------------------------------------------
    // Test 13: Empty input returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_input_errors() {
        let config_vr = VietorisRipsConfig::default();
        let result = vietoris_rips_persistence(&[], 0, 2, &config_vr);
        assert!(
            result.is_err(),
            "empty input should error for VR persistence"
        );

        let config_m = MapperConfig::default();
        let result_m = mapper(&[], 0, 2, &config_m);
        assert!(result_m.is_err(), "empty input should error for mapper");
    }

    // -----------------------------------------------------------------------
    // Test 14: PersistencePair::persistence() and is_essential()
    // -----------------------------------------------------------------------
    #[test]
    fn test_persistence_pair_methods() {
        let finite = PersistencePair {
            birth: 1.0,
            death: 3.0,
            dimension: 0,
        };
        assert!((finite.persistence() - 2.0).abs() < 1e-12);
        assert!(!finite.is_essential());

        let essential = PersistencePair {
            birth: 0.5,
            death: f64::INFINITY,
            dimension: 1,
        };
        assert!(essential.persistence().is_infinite());
        assert!(essential.is_essential());
    }

    // -----------------------------------------------------------------------
    // Test 15: H0-only mode produces no H1 pairs
    // -----------------------------------------------------------------------
    #[test]
    fn test_h0_only_mode() {
        let (data, n, dim) = triangle_data();
        let config = VietorisRipsConfig {
            max_radius: 2.0,
            max_dimension: 0,
            n_points: n,
        };
        let diag = vietoris_rips_persistence(&data, n, dim, &config).expect("H0 only");
        let h1_count = diag.pairs.iter().filter(|p| p.dimension == 1).count();
        assert_eq!(h1_count, 0, "H0-only mode must not produce H1 pairs");
        assert_eq!(diag.betti_1, 0, "H0-only mode: betti_1 = 0");
    }

    // -----------------------------------------------------------------------
    // Test 16: Shape mismatch returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_shape_mismatch_error() {
        let data = vec![1.0_f64, 2.0, 3.0]; // 3 elements, but claimed n=2, dim=2 → needs 4
        let config = VietorisRipsConfig {
            max_radius: 1.0,
            max_dimension: 1,
            n_points: 2,
        };
        let result = vietoris_rips_persistence(&data, 2, 2, &config);
        assert!(result.is_err(), "shape mismatch should return error");
    }

    // -----------------------------------------------------------------------
    // Test 17: MapperGraph adjacency list is consistent
    // -----------------------------------------------------------------------
    #[test]
    fn test_mapper_adjacency_consistent() {
        let data = vec![0.0_f64, 0.0, 0.5, 0.0, 1.0, 0.0, 1.5, 0.0, 2.0, 0.0];
        let config = MapperConfig {
            n_intervals: 4,
            overlap: 0.5,
            n_clusters: 1,
            seed: 0,
        };
        let graph = mapper(&data, 5, 2, &config).expect("line mapper");
        let adj = graph.adjacency_list();
        // Every edge (u,v) should appear as v in adj[u] and u in adj[v].
        for &(u, v) in &graph.edges {
            assert!(adj[u].contains(&v), "adj list missing v={v} from u={u}");
            assert!(adj[v].contains(&u), "adj list missing u={u} from v={v}");
        }
    }
}
