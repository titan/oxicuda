//! Ball Mapper (Dlotko 2019): covers data with fixed-radius balls around landmarks.
//!
//! Each data point is assigned to every ball whose landmark lies within radius ε.
//! The output graph has nodes = non-empty balls and edges = ball pairs sharing ≥ 1 point.
//! Landmark selection strategies: AllPoints, Random(k), MaxMin farthest-point sampling.

use std::collections::HashSet;

use crate::error::{TdaError, TdaResult};
use crate::handle::LcgRng;

/// Strategy for selecting landmark points.
#[derive(Debug, Clone, Copy)]
pub enum LandmarkStrategy {
    /// Every data point becomes a landmark (exact ball cover).
    AllPoints,
    /// Choose `n` landmarks uniformly at random (Fisher-Yates shuffle prefix).
    Random(usize),
    /// Choose `n` landmarks via greedy farthest-point sampling.
    MaxMin(usize),
}

/// Configuration for Ball Mapper.
#[derive(Debug, Clone)]
pub struct BallMapperConfig {
    /// Ball radius ε (must be > 0).
    pub radius: f64,
    /// How to choose landmark points.
    pub landmark_strategy: LandmarkStrategy,
    /// Minimum number of data points for a ball to appear as a node (default 1).
    pub min_points: usize,
}

/// A single node in the Ball Mapper graph.
#[derive(Debug, Clone)]
pub struct BallNode {
    /// Index of the landmark point in the original data array.
    pub landmark_idx: usize,
    /// Coordinates of the landmark (copy, for convenience).
    pub landmark: Vec<f64>,
    /// Indices of data points strictly inside this ball (L2 ≤ radius).
    pub point_indices: Vec<usize>,
    /// Ball radius used when building this node.
    pub radius: f64,
}

/// Result of running Ball Mapper.
#[derive(Debug, Clone)]
pub struct BallMapperResult {
    pub nodes: Vec<BallNode>,
    /// Undirected edges with i < j (node indices into `nodes`).
    pub edges: Vec<(usize, usize)>,
    pub n_points: usize,
}

/// Ball Mapper algorithm.
pub struct BallMapper;

impl BallMapper {
    /// Run Ball Mapper on a row-major point cloud.
    ///
    /// `points`: flat slice of length `n_points * dim`.
    /// `dim`: coordinate dimension of each point.
    pub fn run(
        points: &[f64],
        n_points: usize,
        dim: usize,
        cfg: &BallMapperConfig,
        rng: &mut LcgRng,
    ) -> TdaResult<BallMapperResult> {
        if n_points == 0 {
            return Err(TdaError::EmptyPointCloud);
        }
        if dim == 0 {
            return Err(TdaError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if cfg.radius <= 0.0 || !cfg.radius.is_finite() {
            return Err(TdaError::ParameterOutOfRange(format!(
                "radius must be > 0 and finite, got {}",
                cfg.radius
            )));
        }
        if points.len() != n_points * dim {
            return Err(TdaError::DimensionMismatch {
                expected: n_points * dim,
                got: points.len(),
            });
        }

        // Step 1: select landmarks
        let landmark_indices: Vec<usize> = match cfg.landmark_strategy {
            LandmarkStrategy::AllPoints => (0..n_points).collect(),
            LandmarkStrategy::Random(k) => {
                if k == 0 || k > n_points {
                    return Err(TdaError::ParameterOutOfRange(format!(
                        "Random landmark count {k} out of range [1, {n_points}]"
                    )));
                }
                Self::random_landmarks(n_points, k, rng)
            }
            LandmarkStrategy::MaxMin(k) => {
                if k == 0 || k > n_points {
                    return Err(TdaError::ParameterOutOfRange(format!(
                        "MaxMin landmark count {k} out of range [1, {n_points}]"
                    )));
                }
                Self::maxmin_landmarks(points, n_points, dim, k, rng)
            }
        };

        // Step 2: assign data points to balls
        let memberships = Self::assign_points(points, n_points, dim, &landmark_indices, cfg.radius);

        // Step 3: build nodes, filtering by min_points
        // Keep a mapping from original landmark index → node index
        let mut node_of_landmark: Vec<Option<usize>> = vec![None; landmark_indices.len()];
        let mut nodes: Vec<BallNode> = Vec::new();

        for (li, &lm_idx) in landmark_indices.iter().enumerate() {
            let members = &memberships[li];
            if members.len() >= cfg.min_points.max(1) {
                let start = lm_idx * dim;
                let landmark_coords = points[start..start + dim].to_vec();
                node_of_landmark[li] = Some(nodes.len());
                nodes.push(BallNode {
                    landmark_idx: lm_idx,
                    landmark: landmark_coords,
                    point_indices: members.clone(),
                    radius: cfg.radius,
                });
            }
        }

        // Step 4: build edges between nodes whose point_indices sets overlap
        let edges = Self::build_edges_from_nodes(&nodes);

        Ok(BallMapperResult {
            nodes,
            edges,
            n_points,
        })
    }

    /// Select `k` random landmark indices via partial Fisher-Yates shuffle.
    fn random_landmarks(n_points: usize, k: usize, rng: &mut LcgRng) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n_points).collect();
        for i in 0..k {
            let j = i + rng.next_usize(n_points - i);
            indices.swap(i, j);
        }
        indices[..k].to_vec()
    }

    /// Select `n_landmarks` indices via greedy farthest-point sampling (MaxMin).
    ///
    /// Starts from a random point, then iteratively picks the point farthest from
    /// the current landmark set.
    pub fn maxmin_landmarks(
        points: &[f64],
        n_points: usize,
        dim: usize,
        n_landmarks: usize,
        rng: &mut LcgRng,
    ) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(n_landmarks);
        // Distance of each point to the nearest already-selected landmark
        let mut min_dist = vec![f64::MAX; n_points];

        // Seed: random first point
        let first = rng.next_usize(n_points);
        selected.push(first);

        // Update distances after adding first landmark
        let lm_slice = &points[first * dim..(first + 1) * dim];
        for q in 0..n_points {
            min_dist[q] = Self::l2_dist(lm_slice, &points[q * dim..(q + 1) * dim]);
        }

        while selected.len() < n_landmarks {
            // Farthest point from current landmark set
            let next = min_dist
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            selected.push(next);

            // Update min distances with the new landmark
            let lm_sl = &points[next * dim..(next + 1) * dim];
            for q in 0..n_points {
                let d = Self::l2_dist(lm_sl, &points[q * dim..(q + 1) * dim]);
                if d < min_dist[q] {
                    min_dist[q] = d;
                }
            }
        }

        selected
    }

    /// Euclidean (L2) distance between two equal-length slices.
    #[inline]
    pub fn l2_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// For each landmark, find all data-point indices within L2 ≤ radius.
    pub fn assign_points(
        points: &[f64],
        n_points: usize,
        dim: usize,
        landmarks: &[usize],
        radius: f64,
    ) -> Vec<Vec<usize>> {
        landmarks
            .iter()
            .map(|&lm_idx| {
                let lm_slice = &points[lm_idx * dim..(lm_idx + 1) * dim];
                (0..n_points)
                    .filter(|&q| Self::l2_dist(lm_slice, &points[q * dim..(q + 1) * dim]) <= radius)
                    .collect()
            })
            .collect()
    }

    /// Build undirected edges (i < j) between landmark-level memberships.
    ///
    /// Two landmarks are connected iff their assigned-point sets share at least one index.
    pub fn build_edges(memberships: &[Vec<usize>], n_landmarks: usize) -> Vec<(usize, usize)> {
        // For each data point, record which landmarks contain it, then emit edges from
        // every pair in that list.  This avoids O(n_landmarks²) pairwise intersection.
        if n_landmarks == 0 {
            return vec![];
        }

        // Find the maximum point index to size the per-point landmark lists
        let max_pt = memberships
            .iter()
            .flat_map(|m| m.iter())
            .copied()
            .max()
            .map(|v| v + 1)
            .unwrap_or(0);

        // per_point[p] = list of landmark indices that contain point p
        let mut per_point: Vec<Vec<usize>> = vec![Vec::new(); max_pt];
        for (li, members) in memberships.iter().enumerate() {
            for &pt in members {
                per_point[pt].push(li);
            }
        }

        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        for pt_landmarks in &per_point {
            let m = pt_landmarks.len();
            for a in 0..m {
                for b in (a + 1)..m {
                    let (i, j) = (
                        pt_landmarks[a].min(pt_landmarks[b]),
                        pt_landmarks[a].max(pt_landmarks[b]),
                    );
                    edge_set.insert((i, j));
                }
            }
        }

        let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
        edges.sort_unstable();
        edges
    }

    /// Build edges from already-constructed BallNodes (using node-level indices).
    fn build_edges_from_nodes(nodes: &[BallNode]) -> Vec<(usize, usize)> {
        let n_nodes = nodes.len();
        if n_nodes <= 1 {
            return vec![];
        }

        // Build per-point → node list mapping (using point_indices from each node)
        let max_pt = nodes
            .iter()
            .flat_map(|nd| nd.point_indices.iter())
            .copied()
            .max()
            .map(|v| v + 1)
            .unwrap_or(0);

        let mut per_point: Vec<Vec<usize>> = vec![Vec::new(); max_pt];
        for (ni, nd) in nodes.iter().enumerate() {
            for &pt in &nd.point_indices {
                per_point[pt].push(ni);
            }
        }

        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        for pt_nodes in &per_point {
            let m = pt_nodes.len();
            for a in 0..m {
                for b in (a + 1)..m {
                    let (i, j) = (pt_nodes[a].min(pt_nodes[b]), pt_nodes[a].max(pt_nodes[b]));
                    edge_set.insert((i, j));
                }
            }
        }

        let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
        edges.sort_unstable();
        edges
    }

    /// Fraction of data points covered by at least one ball.
    pub fn coverage(memberships: &[Vec<usize>], n_points: usize) -> f64 {
        if n_points == 0 {
            return 0.0;
        }
        let mut covered = vec![false; n_points];
        for members in memberships {
            for &pt in members {
                if pt < n_points {
                    covered[pt] = true;
                }
            }
        }
        let count = covered.iter().filter(|&&c| c).count();
        count as f64 / n_points as f64
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(12345)
    }

    /// 5 points in 2D on a line: (0,0),(1,0),(2,0),(3,0),(4,0)
    fn line_points() -> Vec<f64> {
        vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]
    }

    fn default_cfg(radius: f64) -> BallMapperConfig {
        BallMapperConfig {
            radius,
            landmark_strategy: LandmarkStrategy::AllPoints,
            min_points: 1,
        }
    }

    // 1. AllPoints strategy: landmark count = n_points, at least one node
    #[test]
    fn ball_mapper_all_points_strategy() {
        let pts = line_points();
        let cfg = default_cfg(1.1);
        let result = BallMapper::run(&pts, 5, 2, &cfg, &mut make_rng()).unwrap();
        assert_eq!(result.nodes.len(), 5);
    }

    // 2. Non-trivial input → at least one node
    #[test]
    fn ball_mapper_output_nodes_nonempty() {
        let pts = line_points();
        let cfg = default_cfg(0.5);
        let result = BallMapper::run(&pts, 5, 2, &cfg, &mut make_rng()).unwrap();
        assert!(!result.nodes.is_empty());
    }

    // 3. All edges are undirected with i < j
    #[test]
    fn ball_mapper_edges_undirected() {
        let pts = line_points();
        let cfg = default_cfg(1.5);
        let result = BallMapper::run(&pts, 5, 2, &cfg, &mut make_rng()).unwrap();
        for &(i, j) in &result.edges {
            assert!(i < j, "edge ({i},{j}) violates i < j");
        }
    }

    // 4. Tiny radius → no edges (each point isolated in its own ball)
    #[test]
    fn ball_mapper_radius_too_small_no_edges() {
        let pts = line_points();
        // radius = 0.01: only the landmark itself is in each ball
        let cfg = default_cfg(0.01);
        let result = BallMapper::run(&pts, 5, 2, &cfg, &mut make_rng()).unwrap();
        assert!(
            result.edges.is_empty(),
            "expected no edges, got {:?}",
            result.edges
        );
    }

    // 5. Huge radius → all nodes connected (complete graph if ≥ 2 nodes)
    #[test]
    fn ball_mapper_large_radius_complete_graph() {
        let pts = line_points();
        let cfg = default_cfg(1000.0);
        let result = BallMapper::run(&pts, 5, 2, &cfg, &mut make_rng()).unwrap();
        let n = result.nodes.len();
        // Complete graph has n*(n-1)/2 edges
        assert_eq!(result.edges.len(), n * (n - 1) / 2);
    }

    // 6. Coverage is in [0, 1]
    #[test]
    fn ball_mapper_coverage_in_range() {
        let pts = line_points();
        let cfg = default_cfg(0.5);
        let result = BallMapper::run(&pts, 5, 2, &cfg, &mut make_rng()).unwrap();
        let memberships: Vec<Vec<usize>> = result
            .nodes
            .iter()
            .map(|n| n.point_indices.clone())
            .collect();
        let cov = BallMapper::coverage(&memberships, 5);
        assert!((0.0..=1.0).contains(&cov), "coverage = {cov}");
    }

    // 7. AllPoints strategy → every point is in at least the ball centered at itself → coverage = 1
    #[test]
    fn ball_mapper_coverage_full_for_all_points() {
        let pts = line_points();
        let cfg = default_cfg(0.5); // each point is at least in its own ball
        let result = BallMapper::run(&pts, 5, 2, &cfg, &mut make_rng()).unwrap();
        let memberships: Vec<Vec<usize>> = result
            .nodes
            .iter()
            .map(|n| n.point_indices.clone())
            .collect();
        let cov = BallMapper::coverage(&memberships, 5);
        assert!((cov - 1.0).abs() < 1e-12, "coverage = {cov}");
    }

    // 8. MaxMin returns exactly n_landmarks indices
    #[test]
    fn maxmin_landmarks_count() {
        let pts = line_points();
        let lm = BallMapper::maxmin_landmarks(&pts, 5, 2, 3, &mut make_rng());
        assert_eq!(lm.len(), 3);
    }

    // 9. MaxMin: all indices are distinct
    #[test]
    fn maxmin_landmarks_unique() {
        let pts = line_points();
        let lm = BallMapper::maxmin_landmarks(&pts, 5, 2, 5, &mut make_rng());
        let unique: std::collections::HashSet<usize> = lm.iter().copied().collect();
        assert_eq!(unique.len(), lm.len());
    }

    // 10. MaxMin: all indices < n_points
    #[test]
    fn maxmin_landmarks_within_bounds() {
        let pts = line_points();
        let lm = BallMapper::maxmin_landmarks(&pts, 5, 2, 4, &mut make_rng());
        for &idx in &lm {
            assert!(idx < 5, "idx {idx} out of bounds");
        }
    }

    // 11. l2_dist self = 0
    #[test]
    fn l2_dist_self_zero() {
        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(BallMapper::l2_dist(&x, &x), 0.0);
    }

    // 12. l2_dist Pythagorean triple
    #[test]
    fn l2_dist_pythagoras() {
        let a = vec![3.0, 0.0];
        let b = vec![0.0, 4.0];
        assert!((BallMapper::l2_dist(&a, &b) - 5.0).abs() < 1e-12);
    }

    // 13. assign_points with radius=0: only the landmark itself is within distance 0
    //     (L2 distance from a point to itself is exactly 0.0 ≤ 0.0 = radius)
    #[test]
    fn assign_points_radius_zero() {
        let pts = line_points();
        // landmark index 0 is point (0.0, 0.0); radius=0 should include only itself
        let members = BallMapper::assign_points(&pts, 5, 2, &[0], 0.0);
        assert_eq!(members[0], vec![0]);
    }

    // 14. build_edges: three overlapping balls sharing points → triangle edges
    #[test]
    fn build_edges_three_points_share() {
        // Memberships: ball0={0,1}, ball1={1,2}, ball2={0,2}  → edges (0,1),(0,2),(1,2)
        let memberships = vec![vec![0, 1], vec![1, 2], vec![0, 2]];
        let edges = BallMapper::build_edges(&memberships, 3);
        assert_eq!(edges.len(), 3);
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(0, 2)));
        assert!(edges.contains(&(1, 2)));
    }

    // 15. Random strategy returns approximately k nodes
    #[test]
    fn ball_mapper_random_strategy() {
        let pts = line_points();
        let cfg = BallMapperConfig {
            radius: 0.5,
            landmark_strategy: LandmarkStrategy::Random(3),
            min_points: 1,
        };
        let result = BallMapper::run(&pts, 5, 2, &cfg, &mut make_rng()).unwrap();
        // With radius=0.5 and 5 collinear points 1 apart, each point only covers itself
        assert_eq!(result.nodes.len(), 3);
    }

    // 16. radius=0 (exactly) → Err ParameterOutOfRange
    #[test]
    fn ball_mapper_err_zero_radius() {
        let pts = line_points();
        let cfg = default_cfg(0.0);
        let err = BallMapper::run(&pts, 5, 2, &cfg, &mut make_rng());
        assert!(err.is_err(), "expected error for radius=0");
    }

    // 17. n_points=0 → Err EmptyPointCloud
    #[test]
    fn ball_mapper_err_empty_points() {
        let pts: Vec<f64> = vec![];
        let cfg = default_cfg(1.0);
        let err = BallMapper::run(&pts, 0, 2, &cfg, &mut make_rng());
        assert!(err.is_err());
    }

    // 18. min_points filter removes singleton balls
    #[test]
    fn ball_mapper_min_points_filter() {
        // 3 points: (0,0),(10,0),(20,0) spaced far apart, radius=0.5
        // each ball covers only its landmark → 1 point per ball
        // With min_points=2, all balls are filtered out
        let pts = vec![0.0, 0.0, 10.0, 0.0, 20.0, 0.0];
        let cfg = BallMapperConfig {
            radius: 0.5,
            landmark_strategy: LandmarkStrategy::AllPoints,
            min_points: 2,
        };
        let result = BallMapper::run(&pts, 3, 2, &cfg, &mut make_rng()).unwrap();
        assert!(
            result.nodes.is_empty(),
            "expected no nodes, got {}",
            result.nodes.len()
        );
    }
}
