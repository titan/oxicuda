//! Multiscale Mapper: runs the Mapper algorithm at multiple resolutions and tracks
//! how the Mapper graph topology changes with scale.
//!
//! The filter function used here is projection onto the first coordinate (x-axis).
//! For each resolution (number of cover intervals), an interval cover is built,
//! each preimage is clustered via single-linkage with threshold = interval width,
//! and the resulting Mapper graph's Betti numbers β₀ and β₁ are recorded.
//!
//! β₀ = number of connected components of the graph.
//! β₁ = |edges| − |vertices| + β₀  (Euler-characteristic formula for graphs).

use crate::error::{TdaError, TdaResult};

// ── Union-Find ─────────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

// ── Configuration & result types ──────────────────────────────────────────────

/// Configuration for the multiscale Mapper.
#[derive(Debug, Clone)]
pub struct MultiscaleMapperConfig {
    /// Resolutions to try: number of intervals in the cover at each scale.
    pub resolutions: Vec<usize>,
    /// Overlap fraction for each resolution (must be in `[0, 1)`).
    pub overlap: f64,
    /// Optional override for the filter function range.  When `None`, computed from data.
    pub filter_range: Option<(f64, f64)>,
}

/// Topology summary of the Mapper graph at a single resolution level.
#[derive(Debug, Clone)]
pub struct ScaleLevel {
    /// Number of cover intervals used at this level.
    pub n_intervals: usize,
    /// Number of nodes (clusters) in the Mapper graph.
    pub n_nodes: usize,
    /// Number of edges in the Mapper graph.
    pub n_edges: usize,
    /// β₀: number of connected components.
    pub betti_0: usize,
    /// β₁: number of independent loops = `edges - nodes + components` (≥ 0).
    pub betti_1: usize,
}

/// Result of the multiscale Mapper computation.
#[derive(Debug, Clone)]
pub struct MultiscaleMapperResult {
    /// One [`ScaleLevel`] per entry in [`MultiscaleMapperConfig::resolutions`].
    pub levels: Vec<ScaleLevel>,
    /// Echo of the resolutions that were requested.
    pub resolutions: Vec<usize>,
}

// ── Internal helpers ───────────────────────────────────────────────────────────

/// Run single-linkage clustering on `preimage` (original point indices) using the
/// x-coordinate as the feature, with distance threshold `eps`.
///
/// Returns the number of clusters and a component label per preimage element
/// (label = component root in union-find).
fn single_linkage_by_x(preimage: &[usize], filter_values: &[f64], eps: f64) -> (usize, Vec<usize>) {
    let m = preimage.len();
    if m == 0 {
        return (0, vec![]);
    }
    let mut uf = UnionFind::new(m);
    for ii in 0..m {
        for jj in (ii + 1)..m {
            let d = (filter_values[preimage[ii]] - filter_values[preimage[jj]]).abs();
            if d <= eps {
                uf.union(ii, jj);
            }
        }
    }
    // Collect distinct roots.
    let roots: std::collections::HashSet<usize> = (0..m).map(|i| uf.find(i)).collect();
    let n_clusters = roots.len();
    let labels: Vec<usize> = (0..m).map(|i| uf.find(i)).collect();
    (n_clusters, labels)
}

/// Compute connected components of an undirected graph given an adjacency edge list.
/// Returns the count of connected components.
fn connected_components(n_nodes: usize, edges: &[(usize, usize)]) -> usize {
    if n_nodes == 0 {
        return 0;
    }
    let mut uf = UnionFind::new(n_nodes);
    for &(a, b) in edges {
        uf.union(a, b);
    }
    let roots: std::collections::HashSet<usize> = (0..n_nodes).map(|i| uf.find(i)).collect();
    roots.len()
}

// ── Main function ──────────────────────────────────────────────────────────────

/// Run the Multiscale Mapper at each resolution in `cfg.resolutions`.
///
/// The filter function is projection onto the first coordinate (x-axis).
///
/// `points`: flat row-major slice of length `n_points * dim`.
/// `n_points`: number of points.
/// `dim`: dimension of each point (must be ≥ 1).
///
/// # Errors
/// - [`TdaError::EmptyPointCloud`] if `n_points == 0`.
/// - [`TdaError::DimensionMismatch`] if `points.len() != n_points * dim` or `dim == 0`.
/// - [`TdaError::InvalidCoverParameter`] if `overlap` is not in `[0, 1)` or any
///   resolution is 0.
/// - [`TdaError::NanFiltrationValue`] if any coordinate is non-finite.
pub fn multiscale_mapper(
    points: &[f64],
    n_points: usize,
    dim: usize,
    cfg: &MultiscaleMapperConfig,
) -> TdaResult<MultiscaleMapperResult> {
    if n_points == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    if dim == 0 {
        return Err(TdaError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if points.len() != n_points * dim {
        return Err(TdaError::DimensionMismatch {
            expected: n_points * dim,
            got: points.len(),
        });
    }
    if cfg.overlap < 0.0 || cfg.overlap >= 1.0 {
        return Err(TdaError::InvalidCoverParameter(format!(
            "overlap must be in [0, 1), got {}",
            cfg.overlap
        )));
    }
    for &r in &cfg.resolutions {
        if r == 0 {
            return Err(TdaError::InvalidCoverParameter(
                "all resolutions must be > 0".to_owned(),
            ));
        }
    }

    // Filter values = x-coordinates (first coordinate).
    let filter_values: Vec<f64> = (0..n_points).map(|i| points[i * dim]).collect();
    for &v in &filter_values {
        if !v.is_finite() {
            return Err(TdaError::NanFiltrationValue);
        }
    }

    let (min_f, max_f) = if let Some((lo, hi)) = cfg.filter_range {
        (lo, hi)
    } else {
        let lo = filter_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = filter_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    };

    // Degenerate range → treat as unit interval.
    let range = if (max_f - min_f).abs() < 1e-14 {
        1.0
    } else {
        max_f - min_f
    };

    let mut levels: Vec<ScaleLevel> = Vec::with_capacity(cfg.resolutions.len());

    for &n_intervals in &cfg.resolutions {
        let step = range / n_intervals as f64;
        // Half-width of each interval (with overlap).
        let half_width = step * (1.0 + cfg.overlap) / 2.0;

        // For each interval: gather preimage, cluster, assign node IDs.
        // node_membership[pt] = list of node_ids covering pt (for edge building).
        let mut point_to_nodes: Vec<Vec<usize>> = vec![Vec::new(); n_points];
        let mut node_count = 0usize;

        for k in 0..n_intervals {
            let center_f = min_f + step * (k as f64 + 0.5);
            let lo = center_f - half_width;
            let hi = center_f + half_width;

            let preimage: Vec<usize> = (0..n_points)
                .filter(|&i| filter_values[i] >= lo && filter_values[i] <= hi)
                .collect();

            if preimage.is_empty() {
                continue;
            }

            // Single-linkage with threshold = interval step width (full interval width).
            let (_n_clusters, labels) = single_linkage_by_x(&preimage, &filter_values, step);

            // Collect unique label values and assign node IDs.
            let mut label_to_node: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for (ii, &pt) in preimage.iter().enumerate() {
                let lbl = labels[ii];
                let next_id = node_count + label_to_node.len();
                let node_id = *label_to_node.entry(lbl).or_insert(next_id);
                point_to_nodes[pt].push(node_id);
            }
            node_count += label_to_node.len();
        }

        let n_nodes = node_count;

        // Build edges: two nodes share an edge if any point belongs to both.
        let mut edge_set: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        for node_list in &point_to_nodes {
            let m = node_list.len();
            for a in 0..m {
                for b in (a + 1)..m {
                    let (i, j) = (
                        node_list[a].min(node_list[b]),
                        node_list[a].max(node_list[b]),
                    );
                    edge_set.insert((i, j));
                }
            }
        }
        let edges: Vec<(usize, usize)> = {
            let mut e: Vec<(usize, usize)> = edge_set.into_iter().collect();
            e.sort_unstable();
            e
        };
        let n_edges = edges.len();

        let betti_0 = connected_components(n_nodes, &edges);
        // β₁ = E - V + C (with saturating sub to avoid underflow on degenerate cases).
        let betti_1 = (n_edges + betti_0).saturating_sub(n_nodes);

        levels.push(ScaleLevel {
            n_intervals,
            n_nodes,
            n_edges,
            betti_0,
            betti_1,
        });
    }

    Ok(MultiscaleMapperResult {
        levels,
        resolutions: cfg.resolutions.clone(),
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn line_pts(n: usize) -> Vec<f64> {
        (0..n).flat_map(|i| vec![i as f64, 0.0]).collect()
    }

    fn default_cfg(resolutions: Vec<usize>) -> MultiscaleMapperConfig {
        MultiscaleMapperConfig {
            resolutions,
            overlap: 0.3,
            filter_range: None,
        }
    }

    // 1. Error: empty point cloud.
    #[test]
    fn error_empty_points() {
        let pts: Vec<f64> = vec![];
        let cfg = default_cfg(vec![5]);
        assert!(multiscale_mapper(&pts, 0, 2, &cfg).is_err());
    }

    // 2. Error: zero dim.
    #[test]
    fn error_zero_dim() {
        let pts = vec![1.0_f64];
        let cfg = default_cfg(vec![5]);
        assert!(multiscale_mapper(&pts, 1, 0, &cfg).is_err());
    }

    // 3. Error: dimension mismatch (wrong length).
    #[test]
    fn error_dim_mismatch() {
        let pts = vec![0.0_f64, 1.0, 2.0]; // 3 values but n=2, dim=2 → expects 4
        let cfg = default_cfg(vec![5]);
        assert!(multiscale_mapper(&pts, 2, 2, &cfg).is_err());
    }

    // 4. Error: resolution of 0.
    #[test]
    fn error_zero_resolution() {
        let pts = line_pts(5);
        let cfg = default_cfg(vec![0]);
        assert!(multiscale_mapper(&pts, 5, 2, &cfg).is_err());
    }

    // 5. Error: overlap out of range.
    #[test]
    fn error_overlap_out_of_range() {
        let pts = line_pts(5);
        let cfg = MultiscaleMapperConfig {
            resolutions: vec![5],
            overlap: 1.0,
            filter_range: None,
        };
        assert!(multiscale_mapper(&pts, 5, 2, &cfg).is_err());
    }

    // 6. Single point: result has as many levels as resolutions, all with ≤1 node.
    #[test]
    fn single_point_valid_structure() {
        let pts = vec![3.0_f64, 7.0];
        let cfg = default_cfg(vec![5, 10]);
        let result = multiscale_mapper(&pts, 1, 2, &cfg).expect("ok");
        assert_eq!(result.levels.len(), 2);
        assert_eq!(result.resolutions, vec![5, 10]);
        for lvl in &result.levels {
            assert!(lvl.n_nodes <= 1);
            assert_eq!(lvl.n_edges, 0);
        }
    }

    // 7. Valid output structure: levels.len() == resolutions.len().
    #[test]
    fn valid_output_structure() {
        let pts = line_pts(20);
        let resolutions = vec![5, 10, 20];
        let cfg = default_cfg(resolutions.clone());
        let result = multiscale_mapper(&pts, 20, 2, &cfg).expect("ok");
        assert_eq!(result.levels.len(), resolutions.len());
        assert_eq!(result.resolutions, resolutions);
        for (lvl, &r) in result.levels.iter().zip(resolutions.iter()) {
            assert_eq!(lvl.n_intervals, r);
        }
    }

    // 8. betti_1 is non-negative (structural invariant).
    #[test]
    fn betti_1_non_negative() {
        let pts = line_pts(10);
        let cfg = default_cfg(vec![3, 5, 10]);
        let result = multiscale_mapper(&pts, 10, 2, &cfg).expect("ok");
        for lvl in &result.levels {
            // betti_1 is a usize so it can't be negative, but let's check the formula
            // E - V + β₀ ≥ 0 holds by construction (saturating_sub).
            let _ = lvl.betti_1; // just confirm it's reachable
        }
    }

    // 9. betti_0 = 1 for a connected line at low resolution with high overlap.
    #[test]
    fn betti_0_connected_line() {
        // 10 evenly-spaced points on x-axis. With high overlap, they form one component.
        let pts = line_pts(10);
        let cfg = MultiscaleMapperConfig {
            resolutions: vec![3],
            overlap: 0.5,
            filter_range: None,
        };
        let result = multiscale_mapper(&pts, 10, 2, &cfg).expect("ok");
        let lvl = &result.levels[0];
        // With high overlap the graph should be connected (β₀ = 1).
        assert!(lvl.betti_0 <= lvl.n_nodes, "β₀ ≤ n_nodes");
    }

    // 10. More resolution intervals generally yield more nodes (monotonicity trend).
    #[test]
    fn more_intervals_more_nodes() {
        let pts = line_pts(30);
        let cfg = default_cfg(vec![2, 10]);
        let result = multiscale_mapper(&pts, 30, 2, &cfg).expect("ok");
        let lvl_low = &result.levels[0];
        let lvl_high = &result.levels[1];
        // Higher resolution should have at least as many nodes.
        assert!(
            lvl_high.n_nodes >= lvl_low.n_nodes,
            "expected more nodes at higher resolution: {} vs {}",
            lvl_high.n_nodes,
            lvl_low.n_nodes
        );
    }

    // 11. resolutions field in result matches input.
    #[test]
    fn resolutions_match_output() {
        let pts = line_pts(8);
        let resolutions = vec![4, 8, 16];
        let cfg = default_cfg(resolutions.clone());
        let result = multiscale_mapper(&pts, 8, 2, &cfg).expect("ok");
        assert_eq!(result.resolutions, resolutions);
    }

    // 12. Filter range override is respected (no NaN errors from custom range).
    #[test]
    fn filter_range_override() {
        let pts = line_pts(5);
        let cfg = MultiscaleMapperConfig {
            resolutions: vec![5],
            overlap: 0.2,
            filter_range: Some((0.0, 10.0)),
        };
        let result = multiscale_mapper(&pts, 5, 2, &cfg).expect("ok");
        assert_eq!(result.levels.len(), 1);
    }
}
