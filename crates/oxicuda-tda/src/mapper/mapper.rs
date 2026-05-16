//! Mapper algorithm: topological summarisation via a nerve of clustered preimages.
//!
//! Given a filter function f: X → ℝ and a cover of the range [min_f, max_f] by
//! overlapping intervals, the Mapper algorithm clusters each preimage f⁻¹(I_α)
//! (single-linkage at scale ε) and builds a graph whose nodes are clusters and
//! edges connect clusters from different intervals that share at least one data point.

use crate::error::{TdaError, TdaResult};

/// Configuration for the Mapper algorithm.
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// Number of cover intervals for the filter range.
    pub n_intervals: usize,
    /// Overlap fraction between consecutive intervals (e.g. 0.5).
    pub overlap_frac: f64,
    /// Single-linkage clustering distance threshold.
    pub cluster_eps: f64,
    /// Minimum number of points for a cluster to be considered a node.
    pub min_pts: usize,
}

/// A node in the Mapper graph — one cluster within one cover interval.
#[derive(Debug, Clone)]
pub struct MapperNode {
    /// Indices into the original point cloud for data points in this cluster.
    pub point_indices: Vec<usize>,
    /// Global cluster identifier (unique across the whole graph).
    pub cluster_id: usize,
    /// Index of the cover interval this cluster came from.
    pub interval_id: usize,
    /// Centroid of the points in this cluster.
    pub center: Vec<f64>,
}

/// Mapper graph: a set of nodes and undirected edges.
///
/// An edge `(i, j)` exists iff node `i` and node `j` share at least one data point
/// (i.e. they come from overlapping intervals and contain a common point).
#[derive(Debug, Clone)]
pub struct MapperGraph {
    pub nodes: Vec<MapperNode>,
    pub edges: Vec<(usize, usize)>,
}

impl MapperGraph {
    /// Number of nodes in the graph.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Degree of node `node` (number of edges incident to it).
    pub fn degree(&self, node: usize) -> usize {
        self.edges
            .iter()
            .filter(|&&(a, b)| a == node || b == node)
            .count()
    }

    /// First Betti number β₁ = |E| - |V| + |connected components|.
    pub fn betti_1(&self) -> usize {
        let comps = self.connected_components().len();
        let v = self.nodes.len();
        let e = self.edges.len();
        (e + comps).saturating_sub(v)
    }

    /// Connected components of the graph via BFS.
    ///
    /// Returns a `Vec` of components, each being a `Vec` of node indices.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.nodes.len();
        if n == 0 {
            return vec![];
        }
        let mut visited = vec![false; n];
        let mut components: Vec<Vec<usize>> = Vec::new();

        // Build adjacency list
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b) in &self.edges {
            if a < n && b < n {
                adj[a].push(b);
                adj[b].push(a);
            }
        }

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut comp = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            while let Some(u) = queue.pop_front() {
                comp.push(u);
                for &w in &adj[u] {
                    if !visited[w] {
                        visited[w] = true;
                        queue.push_back(w);
                    }
                }
            }
            components.push(comp);
        }
        components
    }
}

/// Build a Mapper graph from a point cloud.
///
/// Steps:
/// 1. Apply `filter_fn` to each point to get a scalar value.
/// 2. Divide [min_f, max_f] into `cfg.n_intervals` overlapping intervals.
/// 3. For each interval: gather the preimage points, run single-linkage clustering
///    at scale `cfg.cluster_eps`, and create one `MapperNode` per cluster with ≥ min_pts points.
/// 4. Build edges between nodes from different intervals that share at least one data point.
pub fn build_mapper<F: Fn(&[f64]) -> f64>(
    points: &[f64],
    n_pts: usize,
    n_dims: usize,
    filter_fn: F,
    cfg: &MapperConfig,
) -> TdaResult<MapperGraph> {
    if n_pts == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    if cfg.n_intervals == 0 {
        return Err(TdaError::InvalidCoverParameter(
            "n_intervals must be > 0".to_owned(),
        ));
    }
    if cfg.overlap_frac < 0.0 || cfg.overlap_frac >= 1.0 {
        return Err(TdaError::InvalidCoverParameter(format!(
            "overlap_frac must be in [0, 1), got {}",
            cfg.overlap_frac
        )));
    }

    // Step 1: compute filter values
    let filter_values: Vec<f64> = (0..n_pts)
        .map(|i| filter_fn(&points[i * n_dims..(i + 1) * n_dims]))
        .collect();

    let min_f = filter_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_f = filter_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if !min_f.is_finite() || !max_f.is_finite() {
        return Err(TdaError::NanFiltrationValue);
    }

    // Degenerate case: all points have the same filter value
    let range = if (max_f - min_f).abs() < 1e-14 {
        1.0
    } else {
        max_f - min_f
    };

    let n_int = cfg.n_intervals;
    let step = range / n_int as f64;
    let half_width = step * (1.0 + cfg.overlap_frac) / 2.0; // half-width with overlap

    let mut nodes: Vec<MapperNode> = Vec::new();
    let mut cluster_id_counter = 0usize;

    // For each interval
    for k in 0..n_int {
        let center_f = min_f + step * (k as f64 + 0.5);
        let lo = center_f - half_width;
        let hi = center_f + half_width;

        // Preimage: points whose filter value falls in [lo, hi]
        let preimage: Vec<usize> = (0..n_pts)
            .filter(|&i| filter_values[i] >= lo && filter_values[i] <= hi)
            .collect();

        if preimage.is_empty() {
            continue;
        }

        // Single-linkage clustering via Union-Find on the preimage
        let clusters = single_linkage_clusters(&preimage, points, n_dims, cfg.cluster_eps);

        // Create nodes for clusters with enough points
        for cluster_pts in clusters {
            if cluster_pts.len() < cfg.min_pts {
                continue;
            }
            // Compute centroid
            let mut center = vec![0.0_f64; n_dims];
            for &pi in &cluster_pts {
                for (d, c) in center.iter_mut().enumerate() {
                    *c += points[pi * n_dims + d];
                }
            }
            let inv_n = 1.0 / cluster_pts.len() as f64;
            for c in center.iter_mut() {
                *c *= inv_n;
            }
            nodes.push(MapperNode {
                point_indices: cluster_pts,
                cluster_id: cluster_id_counter,
                interval_id: k,
                center,
            });
            cluster_id_counter += 1;
        }
    }

    // Build edges: two nodes share an edge if they share at least one data point
    let n_nodes = nodes.len();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            // Check for shared points
            let shared = nodes[i]
                .point_indices
                .iter()
                .any(|p| nodes[j].point_indices.contains(p));
            if shared {
                edges.push((i, j));
            }
        }
    }

    Ok(MapperGraph { nodes, edges })
}

/// Single-linkage clustering on a subset of points.
///
/// Builds a graph where `preimage[i]` and `preimage[j]` are connected if their Euclidean
/// distance is ≤ `eps`, then returns connected components.
fn single_linkage_clusters(
    preimage: &[usize],
    points: &[f64],
    n_dims: usize,
    eps: f64,
) -> Vec<Vec<usize>> {
    let m = preimage.len();
    if m == 0 {
        return vec![];
    }

    // Union-Find
    let mut parent: Vec<usize> = (0..m).collect();
    let mut rank: Vec<usize> = vec![0; m];

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut Vec<usize>, rank: &mut [usize], x: usize, y: usize) {
        let rx = find(parent, x);
        let ry = find(parent, y);
        if rx == ry {
            return;
        }
        match rank[rx].cmp(&rank[ry]) {
            std::cmp::Ordering::Less => parent[rx] = ry,
            std::cmp::Ordering::Greater => parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                parent[ry] = rx;
                rank[rx] += 1;
            }
        }
    }

    let eps_sq = eps * eps;

    for ii in 0..m {
        for jj in (ii + 1)..m {
            let pi = preimage[ii];
            let pj = preimage[jj];
            let mut dist_sq = 0.0_f64;
            for d in 0..n_dims {
                let diff = points[pi * n_dims + d] - points[pj * n_dims + d];
                dist_sq += diff * diff;
            }
            if dist_sq <= eps_sq {
                union(&mut parent, &mut rank, ii, jj);
            }
        }
    }

    // Collect components
    let mut component_map: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (ii, &pt) in preimage.iter().enumerate() {
        let root = find(&mut parent, ii);
        component_map.entry(root).or_default().push(pt);
    }
    component_map.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapper_two_clusters() {
        // Two well-separated clusters: x in [0,1] and x in [10,11]
        let pts: Vec<f64> = (0..10)
            .map(|i| i as f64 / 10.0)
            .chain((0..10).map(|i| 10.0 + i as f64 / 10.0))
            .flat_map(|x| vec![x, 0.0])
            .collect();
        let n = 20;
        let cfg = MapperConfig {
            n_intervals: 4,
            overlap_frac: 0.3,
            cluster_eps: 2.0,
            min_pts: 1,
        };
        let graph = build_mapper(&pts, n, 2, |p| p[0], &cfg).expect("ok");
        assert!(graph.n_nodes() >= 2);
    }
}
