//! ES-HyperNEAT: Evolvable Substrate HyperNEAT.
//!
//! Reference: Risi & Stanley, "An Enhanced Hypercube-Based Encoding for Evolving the
//! Placement, Density, and Connectivity of Neurons", Artificial Life 18(4):331-363, 2012.
//!
//! ES-HyperNEAT extends HyperNEAT by automatically discovering the placement and
//! connectivity of hidden substrate neurons from CPPN output geometry, rather than
//! requiring the user to specify a fixed hidden-layer geometry.  The CPPN is queried
//! on a resolution×resolution grid; high-magnitude regions indicate where hidden
//! neurons should be placed.

#![allow(clippy::needless_range_loop)]

use super::hyperneat::{
    CppnConfig, CppnWeights, Substrate, cppn_forward_pub as cppn_forward, hyperneat_query_weights,
};
use crate::{EvolError, EvolResult, handle::LcgRng};

// ─── ES-HyperNEAT configuration ─────────────────────────────────────────────

/// Configuration for ES-HyperNEAT.
///
/// Extends `HyperNeatConfig` with substrate-discovery parameters.
#[derive(Debug, Clone)]
pub struct EsHyperNeatConfig {
    /// CPPN architecture.
    pub cppn: CppnConfig,
    /// Fixed input and output layer coordinates — the base substrate.
    /// The hidden layer is discovered automatically.
    pub input_coords: Vec<(f64, f64)>,
    /// Fixed output coordinates.
    pub output_coords: Vec<(f64, f64)>,
    /// Minimum CPPN response magnitude required to express a connection
    /// (same role as `expression_threshold` in HyperNEAT).
    pub expression_threshold: f64,
    /// Number of probe points per axis in the discovery grid (default 11).
    ///
    /// Retained for backward compatibility and as the grid resolution used by
    /// the legacy grid-probe discovery path; the quadtree discovery instead
    /// uses `initial_depth` / `max_depth`.
    pub resolution: usize,
    /// Minimum CPPN response to place a hidden node at a probed location (default 0.1).
    pub placement_threshold: f64,
    /// Minimum |weight| to keep a connection after substrate query (default 0.05).
    pub prune_threshold: f64,
    /// ES-HyperNEAT quadtree: minimum subdivision depth before the variance
    /// criterion is consulted. The quadtree is always subdivided to at least
    /// this depth, giving an initial resolution of `2^initial_depth` cells per
    /// axis (default 2 → 4×4).
    pub initial_depth: usize,
    /// ES-HyperNEAT quadtree: maximum subdivision depth. Subdivision stops once
    /// this depth is reached regardless of variance (default 4 → up to 16×16).
    pub max_depth: usize,
    /// ES-HyperNEAT quadtree: division variance threshold. A quad keeps
    /// subdividing while the variance of its four children's CPPN weights
    /// exceeds this value (default 0.03).
    pub division_threshold: f64,
    /// ES-HyperNEAT quadtree: band-pruning threshold. A child quad is expressed
    /// as a hidden node when its local band value (the directional weight
    /// variance against its cardinal neighbours) exceeds this value (default 0.3).
    pub band_threshold: f64,
    /// Number of (μ+λ)-ES generations.
    pub n_evol_iters: usize,
    /// Initial perturbation standard deviation.
    pub sigma_init: f64,
    /// Per-generation multiplicative decay of sigma.
    pub sigma_decay: f64,
    /// Random seed.
    pub seed: u64,
}

impl EsHyperNeatConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns `InvalidParameter` for any out-of-range value.
    pub fn validate(&self) -> EvolResult<()> {
        if self.input_coords.is_empty() {
            return Err(EvolError::InvalidParameter(
                "EsHyperNeatConfig: input_coords must not be empty".into(),
            ));
        }
        if self.output_coords.is_empty() {
            return Err(EvolError::InvalidParameter(
                "EsHyperNeatConfig: output_coords must not be empty".into(),
            ));
        }
        if self.resolution < 2 {
            return Err(EvolError::InvalidParameter(
                "EsHyperNeatConfig: resolution must be >= 2".into(),
            ));
        }
        if self.expression_threshold < 0.0 {
            return Err(EvolError::InvalidParameter(
                "expression_threshold must be >= 0".into(),
            ));
        }
        if self.placement_threshold < 0.0 {
            return Err(EvolError::InvalidParameter(
                "placement_threshold must be >= 0".into(),
            ));
        }
        if self.prune_threshold < 0.0 {
            return Err(EvolError::InvalidParameter(
                "prune_threshold must be >= 0".into(),
            ));
        }
        if self.max_depth == 0 {
            return Err(EvolError::InvalidParameter(
                "EsHyperNeatConfig: max_depth must be >= 1".into(),
            ));
        }
        if self.initial_depth > self.max_depth {
            return Err(EvolError::InvalidParameter(
                "EsHyperNeatConfig: initial_depth must be <= max_depth".into(),
            ));
        }
        if self.division_threshold < 0.0 {
            return Err(EvolError::InvalidParameter(
                "division_threshold must be >= 0".into(),
            ));
        }
        if self.band_threshold < 0.0 {
            return Err(EvolError::InvalidParameter(
                "band_threshold must be >= 0".into(),
            ));
        }
        if self.n_evol_iters == 0 {
            return Err(EvolError::InvalidParameter(
                "n_evol_iters must be >= 1".into(),
            ));
        }
        if self.sigma_init <= 0.0 {
            return Err(EvolError::InvalidParameter("sigma_init must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.sigma_decay) {
            return Err(EvolError::InvalidParameter(
                "sigma_decay must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }

    /// Build a default configuration for quick tests.
    ///
    /// Uses 3 inputs at y=-0.5, 2 outputs at y=0.5, resolution 7.
    pub fn default_small(cppn: CppnConfig) -> Self {
        let n_in = 3usize;
        let n_out = 2usize;
        let input_coords: Vec<(f64, f64)> = (0..n_in)
            .map(|i| (-1.0 + 2.0 * i as f64 / (n_in - 1) as f64, -0.5))
            .collect();
        let output_coords: Vec<(f64, f64)> = (0..n_out)
            .map(|i| (-1.0 + 2.0 * i as f64 / (n_out - 1) as f64, 0.5))
            .collect();
        Self {
            cppn,
            input_coords,
            output_coords,
            expression_threshold: 0.1,
            resolution: 7,
            placement_threshold: 0.1,
            prune_threshold: 0.05,
            initial_depth: 2,
            max_depth: 4,
            division_threshold: 0.03,
            band_threshold: 0.3,
            n_evol_iters: 10,
            sigma_init: 0.5,
            sigma_decay: 0.95,
            seed: 0,
        }
    }
}

// ─── Discovered substrate wrapper ────────────────────────────────────────────

/// A substrate discovered by ES-HyperNEAT.
///
/// Unlike the fixed `Substrate`, the hidden layer is populated by the
/// discovery algorithm rather than the user.
#[derive(Debug, Clone)]
pub struct EsSubstrate {
    /// Input coordinates (fixed, copied from config).
    pub input_coords: Vec<(f64, f64)>,
    /// Hidden coordinates discovered from the CPPN geometry.
    pub hidden_coords: Vec<(f64, f64)>,
    /// Output coordinates (fixed, copied from config).
    pub output_coords: Vec<(f64, f64)>,
}

impl EsSubstrate {
    /// Convert to a plain `Substrate` for use with standard HyperNEAT helpers.
    ///
    /// If hidden_coords is empty (nothing was discovered), inserts a single
    /// dummy node at the origin so downstream operations remain valid.
    #[must_use]
    pub fn to_substrate(&self) -> Substrate {
        let hidden = if self.hidden_coords.is_empty() {
            vec![(0.0, 0.0)]
        } else {
            self.hidden_coords.clone()
        };
        Substrate {
            input_coords: self.input_coords.clone(),
            hidden_coords: hidden,
            output_coords: self.output_coords.clone(),
        }
    }

    /// Total connection count (input→hidden + hidden→output).
    #[must_use]
    pub fn n_weights(&self) -> usize {
        let n_hid = if self.hidden_coords.is_empty() {
            1
        } else {
            self.hidden_coords.len()
        };
        self.input_coords.len() * n_hid + n_hid * self.output_coords.len()
    }
}

// ─── ES-HyperNEAT state ────────────────────────────────────────────────────

/// State produced by an ES-HyperNEAT run.
#[derive(Debug, Clone)]
pub struct EsHyperNeatState {
    /// Best CPPN weights found during evolution.
    pub cppn_weights: CppnWeights,
    /// Adaptive substrate discovered from the best CPPN.
    pub discovered_substrate: EsSubstrate,
    /// Weight matrix for the discovered substrate network.
    pub substrate_weights: Vec<f64>,
    /// Best fitness seen during evolution.
    pub best_fitness: f64,
    /// Number of generations completed.
    pub generation: usize,
}

// ─── ES-HyperNEAT quadtree (Risi & Stanley 2012) ─────────────────────────────

/// A node of the ES-HyperNEAT division quadtree.
///
/// Each `QuadPoint` covers a square region of the 2-D substrate centred on
/// `(x, y)` with half-side `width` (so the region spans `[x-width, x+width] ×
/// [y-width, y+width]`).  `weight` is the CPPN response sampled at the centre,
/// `level` is the subdivision depth (root = 0), and `children` (when present)
/// are the four sub-quadrants in NW, NE, SW, SE order.
#[derive(Debug, Clone)]
struct QuadPoint {
    x: f64,
    y: f64,
    width: f64,
    weight: f64,
    level: usize,
    children: Vec<QuadPoint>,
}

impl QuadPoint {
    fn new(x: f64, y: f64, width: f64, weight: f64, level: usize) -> Self {
        Self {
            x,
            y,
            width,
            weight,
            level,
            children: Vec::new(),
        }
    }
}

/// Direction in which the CPPN weight is queried while building the quadtree.
///
/// In ES-HyperNEAT the hidden layer is discovered twice: once treating an input
/// neuron as the fixed source and the candidate as the moving target
/// (`OutgoingFromAnchor`), and once treating an output neuron as the fixed
/// target and the candidate as the moving source (`IncomingToAnchor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryDirection {
    /// Anchor is the source; the candidate `(x, y)` is the target.
    OutgoingFromAnchor,
    /// Anchor is the target; the candidate `(x, y)` is the source.
    IncomingToAnchor,
}

/// Sample the CPPN weight for a candidate point relative to a fixed anchor.
///
/// `anchor` is the fixed input (or output) neuron; `(x, y)` is the variable
/// substrate location the quadtree is probing.
#[inline]
fn query_point(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    anchor: (f64, f64),
    x: f64,
    y: f64,
    dir: QueryDirection,
) -> f64 {
    match dir {
        QueryDirection::OutgoingFromAnchor => {
            cppn_forward(cppn, cppn_cfg, anchor.0, anchor.1, x, y)
        }
        QueryDirection::IncomingToAnchor => cppn_forward(cppn, cppn_cfg, x, y, anchor.0, anchor.1),
    }
}

/// Population variance of a slice of CPPN weights.
///
/// Returns `0.0` for an empty slice. Used both as the division criterion
/// (variance among a quad's four children) and inside band-pruning.
fn weight_variance(weights: &[f64]) -> f64 {
    let n = weights.len();
    if n == 0 {
        return 0.0;
    }
    let mean = weights.iter().sum::<f64>() / n as f64;
    weights
        .iter()
        .map(|&w| (w - mean) * (w - mean))
        .sum::<f64>()
        / n as f64
}

/// ES-HyperNEAT **Division & Initialisation**.
///
/// Recursively subdivides the square substrate region rooted at the centre into
/// a quadtree.  At every quad the CPPN is queried at the centre; a quad's four
/// children are created (NW, NE, SW, SE) and the quad keeps subdividing while
///
/// * its `level` is below `initial_depth` (unconditional subdivision down to the
///   initial resolution of `2^initial_depth` cells per axis), **or**
/// * its `level` is below `max_depth` **and** the variance of its four
///   children's CPPN weights exceeds `division_threshold`.
///
/// The returned root owns the whole tree.
fn division_and_initialisation(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    anchor: (f64, f64),
    dir: QueryDirection,
    center_x: f64,
    center_y: f64,
    half_extent: f64,
    initial_depth: usize,
    max_depth: usize,
    division_threshold: f64,
) -> QuadPoint {
    let root_w = query_point(cppn, cppn_cfg, anchor, center_x, center_y, dir);
    let mut root = QuadPoint::new(center_x, center_y, half_extent, root_w, 0);

    // Iterative breadth-first construction over indices into a flat arena would
    // require post-hoc tree assembly; instead we build recursively via an inner
    // helper that returns fully-populated subtrees.
    fn build(
        cppn: &CppnWeights,
        cppn_cfg: &CppnConfig,
        anchor: (f64, f64),
        dir: QueryDirection,
        node: &mut QuadPoint,
        initial_depth: usize,
        max_depth: usize,
        division_threshold: f64,
    ) {
        // Never subdivide beyond the maximum depth.
        if node.level >= max_depth {
            return;
        }
        let child_width = node.width / 2.0;
        let child_level = node.level + 1;
        // Child centre offsets: NW, NE, SW, SE.
        let offsets = [
            (-child_width, child_width),  // NW
            (child_width, child_width),   // NE
            (-child_width, -child_width), // SW
            (child_width, -child_width),  // SE
        ];
        let mut children: Vec<QuadPoint> = offsets
            .iter()
            .map(|&(dx, dy)| {
                let cx = node.x + dx;
                let cy = node.y + dy;
                let w = query_point(cppn, cppn_cfg, anchor, cx, cy, dir);
                QuadPoint::new(cx, cy, child_width, w, child_level)
            })
            .collect();

        let child_weights: Vec<f64> = children.iter().map(|c| c.weight).collect();
        let var = weight_variance(&child_weights);

        // Decide whether to recurse: always while below the initial resolution,
        // then only where the CPPN output is "interesting" (high variance).
        let keep_dividing =
            node.level < initial_depth || (node.level < max_depth && var > division_threshold);

        if keep_dividing {
            for child in children.iter_mut() {
                build(
                    cppn,
                    cppn_cfg,
                    anchor,
                    dir,
                    child,
                    initial_depth,
                    max_depth,
                    division_threshold,
                );
            }
        }
        node.children = children;
    }

    build(
        cppn,
        cppn_cfg,
        anchor,
        dir,
        &mut root,
        initial_depth,
        max_depth,
        division_threshold,
    );
    root
}

/// ES-HyperNEAT **Pruning & Extraction** band value for a quad.
///
/// Measures how much the CPPN weight at the quad centre differs from the weight
/// at the centres of its four cardinal neighbours (one quad-width away, sampled
/// directly from the CPPN).  Following Risi & Stanley, the band value is
///
/// ```text
/// band = max( min(d_left, d_right), min(d_top, d_bottom) )
/// ```
///
/// where each `d_*` is the absolute weight difference to that neighbour.  A high
/// band value marks a point on an information "band" — a sharp transition in the
/// connectivity pattern — which is exactly where a hidden node should sit.
fn band_value(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    anchor: (f64, f64),
    dir: QueryDirection,
    node: &QuadPoint,
) -> f64 {
    let step = node.width * 2.0;
    let w = node.weight;
    let sample = |dx: f64, dy: f64| -> f64 {
        query_point(cppn, cppn_cfg, anchor, node.x + dx, node.y + dy, dir)
    };
    let d_left = (w - sample(-step, 0.0)).abs();
    let d_right = (w - sample(step, 0.0)).abs();
    let d_top = (w - sample(0.0, step)).abs();
    let d_bottom = (w - sample(0.0, -step)).abs();
    d_left.min(d_right).max(d_top.min(d_bottom))
}

/// ES-HyperNEAT **Pruning & Extraction**.
///
/// Depth-first traversal of the quadtree.  A child quad is expressed as a hidden
/// node when:
///
/// * the parent's children show enough variance (the region carries
///   information — variance `> division_threshold`), and
/// * the child's local band value exceeds `band_threshold` (the child sits on a
///   high-variance transition rather than in a flat region).
///
/// Leaf quads (no further subdivision) are also candidates: their band value is
/// evaluated directly.  Expressed coordinates are appended to `out`.
fn prune_and_extract(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    anchor: (f64, f64),
    dir: QueryDirection,
    node: &QuadPoint,
    division_threshold: f64,
    band_threshold: f64,
    out: &mut Vec<(f64, f64)>,
) {
    if node.children.is_empty() {
        // Leaf quad: express directly if it lies on an information band.
        if band_value(cppn, cppn_cfg, anchor, dir, node) > band_threshold {
            out.push((node.x, node.y));
        }
        return;
    }

    let child_weights: Vec<f64> = node.children.iter().map(|c| c.weight).collect();
    let var = weight_variance(&child_weights);

    for child in &node.children {
        if child.children.is_empty() {
            // The child is a leaf of the tree: express it where the local
            // variance is high (information region) and it sits on a band.
            if var > division_threshold
                && band_value(cppn, cppn_cfg, anchor, dir, child) > band_threshold
            {
                out.push((child.x, child.y));
            }
        } else {
            // Internal child: keep descending.
            prune_and_extract(
                cppn,
                cppn_cfg,
                anchor,
                dir,
                child,
                division_threshold,
                band_threshold,
                out,
            );
        }
    }
}

/// Run the full ES-HyperNEAT quadtree (division + pruning) for one anchor and
/// one query direction, returning the expressed hidden-node coordinates.
#[allow(clippy::too_many_arguments)]
fn quadtree_discover_for_anchor(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    anchor: (f64, f64),
    dir: QueryDirection,
    half_extent: f64,
    initial_depth: usize,
    max_depth: usize,
    division_threshold: f64,
    band_threshold: f64,
    out: &mut Vec<(f64, f64)>,
) {
    let root = division_and_initialisation(
        cppn,
        cppn_cfg,
        anchor,
        dir,
        0.0,
        0.0,
        half_extent,
        initial_depth,
        max_depth,
        division_threshold,
    );
    prune_and_extract(
        cppn,
        cppn_cfg,
        anchor,
        dir,
        &root,
        division_threshold,
        band_threshold,
        out,
    );
}

/// Discover hidden-node positions via the ES-HyperNEAT quadtree algorithm.
///
/// This is the real Risi & Stanley (2012) **Evolvable-Substrate** discovery: for
/// each input neuron a quadtree is grown over `[-1, 1]²` (the candidate as the
/// connection *target*), and for each output neuron a quadtree is grown (the
/// candidate as the connection *source*).  Each quadtree is built by
/// *Division & Initialisation* (subdivide where the CPPN output varies) and then
/// *Pruning & Extraction* (keep the high-band "information" points).  The union
/// of expressed points from every quadtree — deduplicated within a tolerance —
/// becomes the hidden layer.
///
/// `half_extent` is the half-side of the substrate square (1.0 → `[-1, 1]²`).
///
/// Returns the discovered hidden-node coordinates (possibly empty for a
/// constant/uniform CPPN, since a flat field has no information bands).
#[allow(clippy::too_many_arguments)]
pub fn discover_hidden_nodes_quadtree(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    input_coords: &[(f64, f64)],
    output_coords: &[(f64, f64)],
    half_extent: f64,
    initial_depth: usize,
    max_depth: usize,
    division_threshold: f64,
    band_threshold: f64,
) -> Vec<(f64, f64)> {
    let mut candidates: Vec<(f64, f64)> = Vec::new();

    // Pass 1: input anchors → candidate targets (input → hidden geometry).
    for &anchor in input_coords {
        quadtree_discover_for_anchor(
            cppn,
            cppn_cfg,
            anchor,
            QueryDirection::OutgoingFromAnchor,
            half_extent,
            initial_depth,
            max_depth,
            division_threshold,
            band_threshold,
            &mut candidates,
        );
    }

    // Pass 2: candidate sources → output anchors (hidden → output geometry).
    for &anchor in output_coords {
        quadtree_discover_for_anchor(
            cppn,
            cppn_cfg,
            anchor,
            QueryDirection::IncomingToAnchor,
            half_extent,
            initial_depth,
            max_depth,
            division_threshold,
            band_threshold,
            &mut candidates,
        );
    }

    // Merge near-duplicate expressions. Use the finest quad width as the merge
    // tolerance so two points from neighbouring leaf quads collapse to one node.
    let merge_radius = half_extent / 2f64.powi(max_depth as i32);
    deduplicate_candidates(candidates, merge_radius)
}

// ─── Hidden-node discovery (legacy grid probe) ──────────────────────────────────

/// Discover hidden-node positions from the CPPN's output geometry (legacy grid).
///
/// The algorithm probes a `resolution × resolution` grid in `[-1, 1]²` and
/// considers two passes:
/// 1. For each input coord (anchor at source) → grid point (candidate target):
///    if |CPPN(input_anchor, grid_pt)| > threshold → candidate.
/// 2. For each grid point (candidate source) → output coord (anchor at target):
///    if |CPPN(grid_pt, output_anchor)| > threshold → candidate.
///
/// Candidate points from both passes are merged and deduplicated by merging
/// any pair of candidates within `2/resolution` Euclidean distance.
///
/// Returns the discovered hidden-node coordinates.
///
/// Prefer [`discover_hidden_nodes_quadtree`] for the real ES-HyperNEAT
/// discovery; this grid variant is retained for comparison and compatibility.
pub fn discover_hidden_nodes(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    input_coords: &[(f64, f64)],
    output_coords: &[(f64, f64)],
    resolution: usize,
    threshold: f64,
) -> Vec<(f64, f64)> {
    // Build the grid of probe points in [-1, 1]²
    let grid: Vec<(f64, f64)> = grid_points(resolution);
    let merge_radius = 2.0 / resolution as f64;

    let mut candidates: Vec<(f64, f64)> = Vec::new();

    // Pass 1: input-anchor → grid candidate (discovering hidden nodes
    // that receive strong signals from the input layer).
    for &(xs, ys) in input_coords {
        for &(xt, yt) in &grid {
            let raw = cppn_forward(cppn, cppn_cfg, xs, ys, xt, yt);
            if raw.abs() > threshold {
                candidates.push((xt, yt));
            }
        }
    }

    // Pass 2: grid candidate → output-anchor (discovering hidden nodes
    // that send strong signals to the output layer).
    for &(xs, ys) in &grid {
        for &(xt, yt) in output_coords {
            let raw = cppn_forward(cppn, cppn_cfg, xs, ys, xt, yt);
            if raw.abs() > threshold {
                candidates.push((xs, ys));
            }
        }
    }

    // Deduplicate by merging positions within merge_radius.
    deduplicate_candidates(candidates, merge_radius)
}

/// Build a flat list of `resolution × resolution` grid points spanning `[-1,1]²`.
fn grid_points(resolution: usize) -> Vec<(f64, f64)> {
    let mut pts = Vec::with_capacity(resolution * resolution);
    for i in 0..resolution {
        let y = if resolution == 1 {
            0.0
        } else {
            -1.0 + 2.0 * i as f64 / (resolution - 1) as f64
        };
        for j in 0..resolution {
            let x = if resolution == 1 {
                0.0
            } else {
                -1.0 + 2.0 * j as f64 / (resolution - 1) as f64
            };
            pts.push((x, y));
        }
    }
    pts
}

/// Merge any two candidates within `radius` Euclidean distance using a
/// greedy scan: for each candidate, if it is close to an already-accepted
/// representative, skip it; otherwise add it as a new representative.
fn deduplicate_candidates(candidates: Vec<(f64, f64)>, radius: f64) -> Vec<(f64, f64)> {
    let mut result: Vec<(f64, f64)> = Vec::new();
    let r2 = radius * radius;
    for (cx, cy) in candidates {
        let too_close = result.iter().any(|&(rx, ry)| {
            let dx = cx - rx;
            let dy = cy - ry;
            dx * dx + dy * dy <= r2
        });
        if !too_close {
            result.push((cx, cy));
        }
    }
    result
}

// ─── Substrate discovery (public API) ────────────────────────────────────────

/// Half-side of the substrate square; all coordinates live in `[-1, 1]²`.
const SUBSTRATE_HALF_EXTENT: f64 = 1.0;

/// Map a probe `resolution` to a quadtree depth: the smallest `d` with
/// `2^d >= resolution - 1`, clamped to `[1, 8]`.
///
/// Used so the legacy `resolution` knob still drives the quadtree path with a
/// comparable cell count when only `resolution` is supplied.
fn resolution_to_depth(resolution: usize) -> usize {
    let target = resolution.saturating_sub(1).max(1);
    let mut depth = 1usize;
    while (1usize << depth) < target && depth < 8 {
        depth += 1;
    }
    depth
}

/// Discover the adaptive substrate from the CPPN and a base (input/output) substrate.
///
/// The hidden layer in `base_substrate` is **ignored** — it is replaced by the
/// set of nodes discovered by the ES-HyperNEAT quadtree (Risi & Stanley 2012)
/// grown from the CPPN's connectivity geometry.
///
/// This convenience entry point derives the quadtree depths from `resolution`
/// (`max_depth = resolution_to_depth(resolution)`, `initial_depth =
/// max_depth - 1`) and uses `placement_threshold` as the band-pruning
/// threshold with a small fixed division threshold.  For full control over the
/// quadtree parameters use [`es_hyperneat_discover_substrate_cfg`].
///
/// # Returns
/// A new `EsSubstrate` with the discovered hidden layer.
pub fn es_hyperneat_discover_substrate(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    base_substrate: &Substrate,
    resolution: usize,
    placement_threshold: f64,
) -> EsSubstrate {
    let max_depth = resolution_to_depth(resolution);
    let initial_depth = max_depth.saturating_sub(1).max(1).min(max_depth);
    let hidden_coords = discover_hidden_nodes_quadtree(
        cppn,
        cppn_cfg,
        &base_substrate.input_coords,
        &base_substrate.output_coords,
        SUBSTRATE_HALF_EXTENT,
        initial_depth,
        max_depth,
        // A modest division threshold keeps subdivision where the field bends;
        // `placement_threshold` doubles as the band-extraction threshold so the
        // existing knob still gates how aggressively nodes are expressed.
        placement_threshold.max(1e-6) * 0.5,
        placement_threshold,
    );
    EsSubstrate {
        input_coords: base_substrate.input_coords.clone(),
        hidden_coords,
        output_coords: base_substrate.output_coords.clone(),
    }
}

/// Discover the adaptive substrate using the full ES-HyperNEAT quadtree
/// configuration (`initial_depth`, `max_depth`, `division_threshold`,
/// `band_threshold`) carried by `cfg`.
///
/// The hidden layer in `base_substrate` is ignored and replaced by the quadtree
/// discovery.  This is the entry point used internally by [`es_hyperneat_run`].
///
/// # Returns
/// A new `EsSubstrate` with the discovered hidden layer.
pub fn es_hyperneat_discover_substrate_cfg(
    cppn: &CppnWeights,
    cfg: &EsHyperNeatConfig,
    base_substrate: &Substrate,
) -> EsSubstrate {
    let hidden_coords = discover_hidden_nodes_quadtree(
        cppn,
        &cfg.cppn,
        &base_substrate.input_coords,
        &base_substrate.output_coords,
        SUBSTRATE_HALF_EXTENT,
        cfg.initial_depth,
        cfg.max_depth,
        cfg.division_threshold,
        cfg.band_threshold,
    );
    EsSubstrate {
        input_coords: base_substrate.input_coords.clone(),
        hidden_coords,
        output_coords: base_substrate.output_coords.clone(),
    }
}

// ─── Substrate weight query with pruning ────────────────────────────────────

/// Query CPPN weights for the ES-discovered substrate and prune weak connections.
///
/// Builds the full weight matrix for the discovered substrate via the standard
/// HyperNEAT geometric query, then zeros out any connection with
/// `|weight| < prune_threshold`.
fn es_query_and_prune(
    cppn: &CppnWeights,
    cppn_cfg: &CppnConfig,
    substrate: &EsSubstrate,
    expression_threshold: f64,
    prune_threshold: f64,
) -> Vec<f64> {
    let plain = substrate.to_substrate();
    let mut weights = hyperneat_query_weights(cppn, cppn_cfg, &plain, expression_threshold);
    for w in weights.iter_mut() {
        if w.abs() < prune_threshold {
            *w = 0.0;
        }
    }
    weights
}

// ─── Forward pass through discovered substrate ───────────────────────────────

/// Run inference through the ES-discovered substrate network.
///
/// Architecture: input → (tanh) hidden → (tanh) output.
///
/// `state.substrate_weights` must match `state.discovered_substrate.n_weights()`.
///
/// # Errors
/// Returns `DimensionMismatch` if `x.len()` does not match the input-layer size.
pub fn es_hyperneat_forward(state: &EsHyperNeatState, x: &[f64]) -> EvolResult<Vec<f64>> {
    let n_in = state.discovered_substrate.input_coords.len();
    if x.len() != n_in {
        return Err(EvolError::DimensionMismatch {
            expected: n_in,
            got: x.len(),
        });
    }

    let plain = state.discovered_substrate.to_substrate();
    let n_hid = plain.hidden_coords.len();
    let n_out = plain.output_coords.len();
    let sw = &state.substrate_weights;

    let expected_len = n_in * n_hid + n_hid * n_out;
    if sw.len() != expected_len {
        return Err(EvolError::DimensionMismatch {
            expected: expected_len,
            got: sw.len(),
        });
    }

    // Input → hidden (tanh)
    let mut hidden = vec![0.0f64; n_hid];
    for h in 0..n_hid {
        let mut pre = 0.0;
        for i in 0..n_in {
            pre += x[i] * sw[i * n_hid + h];
        }
        hidden[h] = pre.tanh();
    }

    // Hidden → output (tanh)
    let ih_end = n_in * n_hid;
    let mut output = vec![0.0f64; n_out];
    for o in 0..n_out {
        let mut pre = 0.0;
        for h in 0..n_hid {
            pre += hidden[h] * sw[ih_end + h * n_out + o];
        }
        output[o] = pre.tanh();
    }

    Ok(output)
}

// ─── (μ+λ)-ES internals ──────────────────────────────────────────────────────

const MU: usize = 5;
const LAMBDA: usize = 20;

/// Perturb a flat parameter vector with Gaussian noise of std `sigma`.
fn perturb(params: &[f64], sigma: f64, rng: &mut LcgRng) -> Vec<f64> {
    params
        .iter()
        .map(|&p| p + rng.next_normal() * sigma)
        .collect()
}

/// Evaluate a flat CPPN parameter vector.
///
/// Returns `(fitness, es_substrate, substrate_weights)`.
fn evaluate_es_params(
    flat: &[f64],
    cfg: &EsHyperNeatConfig,
    fitness_fn: &impl Fn(&[f64], &EsSubstrate) -> f64,
) -> EvolResult<(f64, EsSubstrate, Vec<f64>)> {
    let cppn = CppnWeights::from_flat(flat, cfg.cppn.n_hidden)?;

    // Build a base substrate carrying only the fixed input/output geometry; the
    // hidden layer is discovered by the ES-HyperNEAT quadtree and replaces any
    // hidden coordinates supplied here.
    let base = Substrate {
        input_coords: cfg.input_coords.clone(),
        hidden_coords: Vec::new(),
        output_coords: cfg.output_coords.clone(),
    };

    // Discover the hidden substrate via the quadtree (division + pruning).
    let es_sub = es_hyperneat_discover_substrate_cfg(&cppn, cfg, &base);

    // Query weights with pruning
    let sw = es_query_and_prune(
        &cppn,
        &cfg.cppn,
        &es_sub,
        cfg.expression_threshold,
        cfg.prune_threshold,
    );

    let fitness = fitness_fn(&sw, &es_sub);
    Ok((fitness, es_sub, sw))
}

// ─── Public API: run ES-HyperNEAT ────────────────────────────────────────────

/// Run ES-HyperNEAT via (μ+λ)-ES to optimise CPPN weights and discover the substrate.
///
/// `fitness_fn(substrate_weights, substrate) -> f64` — higher is better.
///
/// At each generation:
/// 1. Generate λ offspring from the μ best parents via Gaussian perturbation.
/// 2. For each offspring: discover the adaptive substrate, query weights, evaluate fitness.
/// 3. Select the top μ individuals (μ+λ selection).
///
/// # Errors
/// Returns `InvalidParameter` if `cfg.validate()` fails.
/// Returns `EmptyPopulation` if the initial population cannot be evaluated.
pub fn es_hyperneat_run(
    fitness_fn: impl Fn(&[f64], &EsSubstrate) -> f64,
    cfg: &EsHyperNeatConfig,
) -> EvolResult<EsHyperNeatState> {
    cfg.validate()?;

    let mut rng = LcgRng::new(cfg.seed);
    let n_params = cfg.cppn.n_params();

    // Initialise μ parents randomly
    let mut parents: Vec<Vec<f64>> = (0..MU)
        .map(|_| {
            (0..n_params)
                .map(|_| rng.next_normal() * cfg.sigma_init)
                .collect()
        })
        .collect();

    // Evaluate initial parents
    let parent_results: Vec<Option<(f64, EsSubstrate, Vec<f64>)>> = parents
        .iter()
        .map(|p| {
            evaluate_es_params(p, cfg, &fitness_fn)
                .ok()
                .filter(|(f, _, _)| f.is_finite())
        })
        .collect();

    // Track best overall
    let best_init_idx = parent_results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.as_ref().map(|(f, _, _)| (i, *f)))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .ok_or(EvolError::EmptyPopulation)?;

    let best_init_fit = parent_results[best_init_idx]
        .as_ref()
        .map(|(f, _, _)| *f)
        .unwrap_or(f64::NEG_INFINITY);

    let mut best_flat = parents[best_init_idx].clone();
    let mut best_fitness = best_init_fit;
    let mut sigma = cfg.sigma_init;

    // Extract parent fitnesses for selection
    let mut parent_fitness: Vec<f64> = parent_results
        .iter()
        .map(|r| r.as_ref().map(|(f, _, _)| *f).unwrap_or(f64::NEG_INFINITY))
        .collect();

    for _gen in 0..cfg.n_evol_iters {
        // Generate λ offspring
        let mut offspring: Vec<(Vec<f64>, f64)> = Vec::with_capacity(LAMBDA);
        for _ in 0..LAMBDA {
            let parent_idx = rng.next_usize(MU);
            let child = perturb(&parents[parent_idx], sigma, &mut rng);
            let fit = evaluate_es_params(&child, cfg, &fitness_fn)
                .ok()
                .filter(|(f, _, _)| f.is_finite())
                .map(|(f, _, _)| f)
                .unwrap_or(f64::NEG_INFINITY);
            offspring.push((child, fit));
        }

        // μ+λ selection
        let mut combined: Vec<(Vec<f64>, f64)> = parents
            .iter()
            .zip(parent_fitness.iter())
            .map(|(p, &f)| (p.clone(), f))
            .collect();
        combined.extend(offspring);
        combined
            .sort_by(|(_, fa), (_, fb)| fb.partial_cmp(fa).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(MU);

        // Update best
        if combined[0].1 > best_fitness {
            best_fitness = combined[0].1;
            best_flat = combined[0].0.clone();
        }

        parents = combined.iter().map(|(p, _)| p.clone()).collect();
        parent_fitness = combined.iter().map(|(_, f)| *f).collect();

        // Sigma annealing
        sigma *= cfg.sigma_decay;
        sigma = sigma.max(1e-8);
    }

    // Final evaluation of the best CPPN to get the discovered substrate + weights
    let best_cppn = CppnWeights::from_flat(&best_flat, cfg.cppn.n_hidden)?;
    let base = Substrate {
        input_coords: cfg.input_coords.clone(),
        hidden_coords: Vec::new(),
        output_coords: cfg.output_coords.clone(),
    };
    let discovered_substrate = es_hyperneat_discover_substrate_cfg(&best_cppn, cfg, &base);
    let substrate_weights = es_query_and_prune(
        &best_cppn,
        &cfg.cppn,
        &discovered_substrate,
        cfg.expression_threshold,
        cfg.prune_threshold,
    );

    Ok(EsHyperNeatState {
        cppn_weights: best_cppn,
        discovered_substrate,
        substrate_weights,
        best_fitness,
        generation: cfg.n_evol_iters,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::hyperneat::{CppnActivation, CppnConfig, CppnWeights};
    use super::*;

    fn default_cppn_cfg() -> CppnConfig {
        CppnConfig::new(
            4,
            vec![
                CppnActivation::Tanh,
                CppnActivation::Gaussian,
                CppnActivation::Sine,
                CppnActivation::Sigmoid,
            ],
        )
        .expect("value should be present")
    }

    fn default_es_cfg() -> EsHyperNeatConfig {
        EsHyperNeatConfig::default_small(default_cppn_cfg())
    }

    type CoordPair = (Vec<(f64, f64)>, Vec<(f64, f64)>);

    fn make_input_output_coords(n_in: usize, n_out: usize) -> CoordPair {
        let input: Vec<(f64, f64)> = (0..n_in)
            .map(|i| {
                let x = if n_in == 1 {
                    0.0
                } else {
                    -1.0 + 2.0 * i as f64 / (n_in - 1) as f64
                };
                (x, -0.5)
            })
            .collect();
        let output: Vec<(f64, f64)> = (0..n_out)
            .map(|i| {
                let x = if n_out == 1 {
                    0.0
                } else {
                    -1.0 + 2.0 * i as f64 / (n_out - 1) as f64
                };
                (x, 0.5)
            })
            .collect();
        (input, output)
    }

    // ── 1: validate catches bad resolution ───────────────────────────────────

    #[test]
    fn validate_bad_resolution() {
        let mut cfg = default_es_cfg();
        cfg.resolution = 1;
        assert!(cfg.validate().is_err());
    }

    // ── 2: validate catches bad sigma_init ───────────────────────────────────

    #[test]
    fn validate_bad_sigma_init() {
        let mut cfg = default_es_cfg();
        cfg.sigma_init = 0.0;
        assert!(cfg.validate().is_err());
    }

    // ── 3: validate catches bad sigma_decay ──────────────────────────────────

    #[test]
    fn validate_bad_sigma_decay() {
        let mut cfg = default_es_cfg();
        cfg.sigma_decay = 1.5;
        assert!(cfg.validate().is_err());
    }

    // ── 4: validate catches empty input_coords ───────────────────────────────

    #[test]
    fn validate_empty_input_coords() {
        let mut cfg = default_es_cfg();
        cfg.input_coords = Vec::new();
        assert!(cfg.validate().is_err());
    }

    // ── 5: validate catches empty output_coords ──────────────────────────────

    #[test]
    fn validate_empty_output_coords() {
        let mut cfg = default_es_cfg();
        cfg.output_coords = Vec::new();
        assert!(cfg.validate().is_err());
    }

    // ── 6: grid_points produces correct count ────────────────────────────────

    #[test]
    fn grid_points_count() {
        let pts = grid_points(5);
        assert_eq!(pts.len(), 25, "5×5 grid should have 25 points");
    }

    // ── 7: grid_points span [-1, 1]² ─────────────────────────────────────────

    #[test]
    fn grid_points_range() {
        let pts = grid_points(7);
        for (x, y) in pts {
            assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&x));
            assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&y));
        }
    }

    // ── 8: deduplicate removes nearby points ─────────────────────────────────

    #[test]
    fn deduplication_works() {
        let candidates = vec![(0.0, 0.0), (0.01, 0.0), (0.9, 0.9), (0.91, 0.9)];
        let result = deduplicate_candidates(candidates, 0.2);
        // The (0.0,0.0) and (0.01,0.0) are within 0.2 → only 1 kept.
        // The (0.9,0.9) and (0.91,0.9) are within 0.2 → only 1 kept.
        assert_eq!(result.len(), 2);
    }

    // ── 9: discover_hidden_nodes returns finite coords ────────────────────────

    #[test]
    fn discover_hidden_nodes_finite() {
        let mut rng = LcgRng::new(42);
        let cppn_cfg = default_cppn_cfg();
        let cppn = CppnWeights::random(cppn_cfg.n_hidden, 1.0, &mut rng);
        let (input_coords, output_coords) = make_input_output_coords(3, 2);
        let hidden = discover_hidden_nodes(&cppn, &cppn_cfg, &input_coords, &output_coords, 7, 0.1);
        for (x, y) in hidden {
            assert!(x.is_finite() && y.is_finite());
        }
    }

    // ── 10: es_hyperneat_discover_substrate produces valid EsSubstrate ────────

    #[test]
    fn discover_substrate_valid() {
        let mut rng = LcgRng::new(77);
        let cppn_cfg = default_cppn_cfg();
        let cppn = CppnWeights::random(cppn_cfg.n_hidden, 0.5, &mut rng);
        let base = Substrate {
            input_coords: vec![(-1.0, -0.5), (0.0, -0.5), (1.0, -0.5)],
            hidden_coords: vec![(0.0, 0.0)],
            output_coords: vec![(-0.5, 0.5), (0.5, 0.5)],
        };
        let es_sub = es_hyperneat_discover_substrate(&cppn, &cppn_cfg, &base, 5, 0.05);
        assert_eq!(es_sub.input_coords, base.input_coords);
        assert_eq!(es_sub.output_coords, base.output_coords);
        // hidden may be empty or non-empty, just check it is finite
        for (x, y) in &es_sub.hidden_coords {
            assert!(x.is_finite() && y.is_finite());
        }
    }

    // ── 11: to_substrate inserts dummy hidden when empty ─────────────────────

    #[test]
    fn to_substrate_empty_hidden_fallback() {
        let es_sub = EsSubstrate {
            input_coords: vec![(0.0, -0.5)],
            hidden_coords: Vec::new(),
            output_coords: vec![(0.0, 0.5)],
        };
        let plain = es_sub.to_substrate();
        assert_eq!(plain.hidden_coords.len(), 1);
        assert_eq!(plain.hidden_coords[0], (0.0, 0.0));
    }

    // ── 12: es_hyperneat_forward output shape and tanh range ─────────────────

    #[test]
    fn es_forward_shape_and_range() {
        let mut rng = LcgRng::new(13);
        let cppn_cfg = default_cppn_cfg();
        let cppn = CppnWeights::random(cppn_cfg.n_hidden, 0.5, &mut rng);
        let base = Substrate {
            input_coords: vec![(-1.0, -0.5), (0.0, -0.5), (1.0, -0.5)],
            hidden_coords: vec![(0.0, 0.0)],
            output_coords: vec![(-0.5, 0.5), (0.5, 0.5)],
        };
        let es_sub = es_hyperneat_discover_substrate(&cppn, &cppn_cfg, &base, 5, 0.05);
        let sw = es_query_and_prune(&cppn, &cppn_cfg, &es_sub, 0.1, 0.05);
        let state = EsHyperNeatState {
            cppn_weights: cppn,
            discovered_substrate: es_sub,
            substrate_weights: sw,
            best_fitness: 0.0,
            generation: 0,
        };
        let x = vec![0.1, -0.2, 0.3];
        let out = es_hyperneat_forward(&state, &x).expect("es_hyperneat_forward should succeed");
        assert_eq!(out.len(), 2);
        for &v in &out {
            assert!(v.abs() <= 1.0 + 1e-10, "output must be in [-1,1]");
        }
    }

    // ── 13: es_hyperneat_forward dimension mismatch error ────────────────────

    #[test]
    fn es_forward_dim_mismatch() {
        let mut rng = LcgRng::new(99);
        let cppn_cfg = default_cppn_cfg();
        let cppn = CppnWeights::random(cppn_cfg.n_hidden, 0.3, &mut rng);
        let base = Substrate {
            input_coords: vec![(-1.0, -0.5), (0.0, -0.5), (1.0, -0.5)],
            hidden_coords: vec![(0.0, 0.0)],
            output_coords: vec![(-0.5, 0.5), (0.5, 0.5)],
        };
        let es_sub = es_hyperneat_discover_substrate(&cppn, &cppn_cfg, &base, 5, 0.05);
        let sw = es_query_and_prune(&cppn, &cppn_cfg, &es_sub, 0.1, 0.05);
        let state = EsHyperNeatState {
            cppn_weights: cppn,
            discovered_substrate: es_sub,
            substrate_weights: sw,
            best_fitness: 0.0,
            generation: 0,
        };
        // Wrong input length (4 instead of 3)
        let result = es_hyperneat_forward(&state, &[0.0, 0.1, 0.2, 0.3]);
        assert!(result.is_err());
    }

    // ── 14: es_hyperneat_run completes without error ──────────────────────────

    #[test]
    fn es_hyperneat_run_completes() {
        let cfg = EsHyperNeatConfig {
            n_evol_iters: 5,
            ..default_es_cfg()
        };
        let state =
            es_hyperneat_run(|_sw, _sub| 1.0, &cfg).expect("es_hyperneat_run should succeed");
        assert_eq!(state.generation, 5);
        assert!(state.best_fitness.is_finite());
    }

    // ── 15: es_hyperneat_run best_fitness is finite ───────────────────────────

    #[test]
    fn es_hyperneat_run_fitness_finite() {
        let cfg = EsHyperNeatConfig {
            n_evol_iters: 8,
            sigma_init: 0.3,
            sigma_decay: 0.9,
            seed: 7,
            ..default_es_cfg()
        };
        let state = es_hyperneat_run(|sw, _sub| -sw.iter().map(|w| w * w).sum::<f64>(), &cfg)
            .expect("value should be present");
        assert!(state.best_fitness.is_finite());
    }

    // ── 16: substrate_weights length matches discovered substrate ─────────────

    #[test]
    fn substrate_weights_length_matches() {
        let cfg = EsHyperNeatConfig {
            n_evol_iters: 3,
            ..default_es_cfg()
        };
        let state =
            es_hyperneat_run(|_sw, _sub| 0.0, &cfg).expect("es_hyperneat_run should succeed");
        let expected = state.discovered_substrate.n_weights();
        assert_eq!(
            state.substrate_weights.len(),
            expected,
            "substrate_weights length must match n_weights"
        );
    }

    // ── 17: prune_threshold zeros out small weights ───────────────────────────

    #[test]
    fn pruning_zeros_small_weights() {
        let mut rng = LcgRng::new(55);
        let cppn_cfg = default_cppn_cfg();
        // Use a CPPN with tiny weights so outputs are small
        let cppn = CppnWeights::random(cppn_cfg.n_hidden, 0.01, &mut rng);
        let es_sub = EsSubstrate {
            input_coords: vec![(-1.0, -0.5), (0.0, -0.5), (1.0, -0.5)],
            hidden_coords: vec![(0.0, 0.0)],
            output_coords: vec![(-0.5, 0.5), (0.5, 0.5)],
        };
        // Set prune_threshold very high to zero everything
        let sw = es_query_and_prune(&cppn, &cppn_cfg, &es_sub, 0.0, 1e9);
        assert!(sw.iter().all(|&w| w == 0.0), "all weights should be pruned");
    }

    // ── Quadtree discovery helpers ────────────────────────────────────────────

    /// CPPN config whose first hidden neuron uses `Sine` — lets us build a
    /// spatially-oscillating weight field for the quadtree to find bands in.
    fn sine_cppn_cfg() -> CppnConfig {
        CppnConfig::new(2, vec![CppnActivation::Sine, CppnActivation::Tanh])
            .expect("cppn cfg should build")
    }

    /// Build a CPPN whose output is `sin(freq * x_tgt)` (a high-frequency
    /// spatial wave in the target x-coordinate). Pass 1 of discovery anchors on
    /// the source and moves the target, so this yields many information bands.
    fn varying_cppn(freq: f64) -> CppnWeights {
        // 2 hidden neurons; only neuron 0 (Sine) is active, weighted on input
        // index 2 = x_tgt. All other weights/biases are zero.
        let mut w = CppnWeights::zeros(2);
        w.hidden_weights[2] = freq; // neuron 0 ← x_tgt (row 0, col 2)
        w.output_weights[0] = 1.0; // read neuron 0 (Sine) directly
        w
    }

    /// Build a CPPN with a constant output of `value` everywhere (no spatial
    /// structure → zero variance → no information bands).
    fn constant_cppn(value: f64) -> CppnWeights {
        let mut w = CppnWeights::zeros(2);
        w.output_bias = value;
        w
    }

    // ── 18: varying CPPN yields MORE than one discovered node ──────────────────

    #[test]
    fn quadtree_varying_cppn_multiple_nodes() {
        let cfg = sine_cppn_cfg();
        let cppn = varying_cppn(6.0);
        let (input_coords, output_coords) = make_input_output_coords(3, 2);
        let hidden = discover_hidden_nodes_quadtree(
            &cppn,
            &cfg,
            &input_coords,
            &output_coords,
            1.0,  // half_extent
            2,    // initial_depth
            5,    // max_depth
            0.02, // division_threshold
            0.2,  // band_threshold
        );
        // A spatial wave must produce many bands → well more than the old
        // placeholder singleton.
        assert!(
            hidden.len() > 1,
            "varying CPPN should discover more than one hidden node, got {}",
            hidden.len()
        );
        // It is not the degenerate placeholder `[(0.0, 0.0)]`.
        assert_ne!(hidden, vec![(0.0, 0.0)]);
    }

    // ── 19: constant CPPN yields few/no extra nodes ───────────────────────────

    #[test]
    fn quadtree_constant_cppn_no_nodes() {
        let cfg = sine_cppn_cfg();
        let cppn = constant_cppn(0.5);
        let (input_coords, output_coords) = make_input_output_coords(3, 2);
        let hidden = discover_hidden_nodes_quadtree(
            &cppn,
            &cfg,
            &input_coords,
            &output_coords,
            1.0,
            2,
            5,
            0.02,
            0.2,
        );
        // A flat field has no information bands → nothing expressed.
        assert!(
            hidden.is_empty(),
            "constant CPPN should discover no hidden nodes, got {}",
            hidden.len()
        );
    }

    // ── 20: varying CPPN discovers strictly more nodes than constant ──────────

    #[test]
    fn quadtree_varying_beats_constant() {
        let cfg = sine_cppn_cfg();
        let (input_coords, output_coords) = make_input_output_coords(3, 2);
        let varying = discover_hidden_nodes_quadtree(
            &varying_cppn(6.0),
            &cfg,
            &input_coords,
            &output_coords,
            1.0,
            2,
            5,
            0.02,
            0.2,
        );
        let constant = discover_hidden_nodes_quadtree(
            &constant_cppn(0.5),
            &cfg,
            &input_coords,
            &output_coords,
            1.0,
            2,
            5,
            0.02,
            0.2,
        );
        assert!(
            varying.len() > constant.len(),
            "varying ({}) must discover more nodes than constant ({})",
            varying.len(),
            constant.len()
        );
    }

    // ── 21: discovered coordinates are within bounds and finite ───────────────

    #[test]
    fn quadtree_nodes_within_bounds_and_finite() {
        let cfg = sine_cppn_cfg();
        let cppn = varying_cppn(8.0);
        let (input_coords, output_coords) = make_input_output_coords(3, 2);
        let hidden = discover_hidden_nodes_quadtree(
            &cppn,
            &cfg,
            &input_coords,
            &output_coords,
            1.0,
            2,
            5,
            0.02,
            0.2,
        );
        assert!(!hidden.is_empty(), "expected discovered nodes");
        for (x, y) in &hidden {
            assert!(x.is_finite() && y.is_finite(), "coords must be finite");
            assert!(
                (-1.0 - 1e-9..=1.0 + 1e-9).contains(x),
                "x = {x} out of substrate bounds"
            );
            assert!(
                (-1.0 - 1e-9..=1.0 + 1e-9).contains(y),
                "y = {y} out of substrate bounds"
            );
        }
    }

    // ── 22: discovered nodes lie in high-variance (nonzero-band) regions ──────

    #[test]
    fn quadtree_nodes_in_high_variance_regions() {
        let cfg = sine_cppn_cfg();
        let freq = 6.0;
        let cppn = varying_cppn(freq);
        let (input_coords, output_coords) = make_input_output_coords(3, 2);
        let max_depth = 5usize;
        let band_threshold = 0.1;
        let hidden = discover_hidden_nodes_quadtree(
            &cppn,
            &cfg,
            &input_coords,
            &output_coords,
            1.0,
            2,
            max_depth,
            0.01,
            band_threshold,
        );
        assert!(!hidden.is_empty(), "expected discovered nodes");
        // Every discovered node must sit where the local band value (the
        // directional variance of the CPPN field) is genuinely high — i.e. on a
        // steep slope of the wave, not on a flat crest. Re-derive the band at the
        // node using the leaf-level quad width and require it to clear the same
        // band threshold the discovery used (allowing a small slack for nodes
        // expressed from a shallower leaf with a slightly larger step).
        let leaf_width = 1.0 / 2f64.powi(max_depth as i32);
        let anchor = input_coords[0];
        for &(x, y) in &hidden {
            let weight = query_point(
                &cppn,
                &cfg,
                anchor,
                x,
                y,
                QueryDirection::OutgoingFromAnchor,
            );
            let probe = QuadPoint::new(x, y, leaf_width, weight, max_depth);
            let band = band_value(
                &cppn,
                &cfg,
                anchor,
                QueryDirection::OutgoingFromAnchor,
                &probe,
            );
            assert!(
                band > band_threshold * 0.5,
                "node ({x},{y}) should sit in a high-band region, got band {band}"
            );
        }
    }

    // ── 23: discovery is deterministic for a fixed CPPN ───────────────────────

    #[test]
    fn quadtree_discovery_deterministic() {
        let cfg = sine_cppn_cfg();
        let cppn = varying_cppn(7.0);
        let (input_coords, output_coords) = make_input_output_coords(3, 2);
        let run = || {
            discover_hidden_nodes_quadtree(
                &cppn,
                &cfg,
                &input_coords,
                &output_coords,
                1.0,
                2,
                5,
                0.02,
                0.2,
            )
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "quadtree discovery must be deterministic");
    }

    // ── 24: division subdivides deeper where the field varies (variance gate) ─

    #[test]
    fn quadtree_division_respects_variance() {
        let cfg = sine_cppn_cfg();
        let anchor = (-1.0, -0.5);
        // High-frequency field → high child variance → deep subdivision.
        let varying_root = division_and_initialisation(
            &varying_cppn(8.0),
            &cfg,
            anchor,
            QueryDirection::OutgoingFromAnchor,
            0.0,
            0.0,
            1.0,
            1,    // initial_depth (shallow forced floor)
            6,    // max_depth
            0.05, // division_threshold
        );
        // Constant field → zero variance → no subdivision past the floor.
        let constant_root = division_and_initialisation(
            &constant_cppn(0.5),
            &cfg,
            anchor,
            QueryDirection::OutgoingFromAnchor,
            0.0,
            0.0,
            1.0,
            1,
            6,
            0.05,
        );

        fn max_depth_of(node: &QuadPoint) -> usize {
            node.children
                .iter()
                .map(max_depth_of)
                .max()
                .unwrap_or(node.level)
        }
        let varying_depth = max_depth_of(&varying_root);
        let constant_depth = max_depth_of(&constant_root);
        assert!(
            varying_depth > constant_depth,
            "varying field (depth {varying_depth}) must subdivide deeper than constant (depth {constant_depth})"
        );
    }

    // ── 25: es_hyperneat_run discovers a consistent (non-placeholder) substrate

    #[test]
    fn es_run_discovers_consistent_substrate() {
        // Reward CPPNs that produce many active substrate weights so evolution
        // is pushed toward structured (band-rich) connectivity patterns.
        let cfg = EsHyperNeatConfig {
            cppn: sine_cppn_cfg(),
            initial_depth: 2,
            max_depth: 5,
            division_threshold: 0.02,
            band_threshold: 0.2,
            n_evol_iters: 6,
            sigma_init: 1.0,
            sigma_decay: 0.9,
            seed: 3,
            ..EsHyperNeatConfig::default_small(sine_cppn_cfg())
        };
        let state = es_hyperneat_run(
            |sw, _sub| sw.iter().filter(|&&w| w != 0.0).count() as f64,
            &cfg,
        )
        .expect("es_hyperneat_run should succeed");
        assert!(state.best_fitness.is_finite());
        // The substrate weight vector must stay consistent with whatever the
        // quadtree discovered as the hidden layer.
        assert_eq!(
            state.substrate_weights.len(),
            state.discovered_substrate.n_weights()
        );
    }

    // ── 26: cfg-driven discovery matches the standalone quadtree call ─────────

    #[test]
    fn discover_substrate_cfg_matches_quadtree() {
        let cfg = EsHyperNeatConfig {
            cppn: sine_cppn_cfg(),
            initial_depth: 2,
            max_depth: 5,
            division_threshold: 0.02,
            band_threshold: 0.2,
            ..EsHyperNeatConfig::default_small(sine_cppn_cfg())
        };
        let cppn = varying_cppn(6.0);
        let base = Substrate {
            input_coords: cfg.input_coords.clone(),
            hidden_coords: Vec::new(),
            output_coords: cfg.output_coords.clone(),
        };
        let via_cfg = es_hyperneat_discover_substrate_cfg(&cppn, &cfg, &base);
        let direct = discover_hidden_nodes_quadtree(
            &cppn,
            &cfg.cppn,
            &cfg.input_coords,
            &cfg.output_coords,
            1.0,
            cfg.initial_depth,
            cfg.max_depth,
            cfg.division_threshold,
            cfg.band_threshold,
        );
        assert_eq!(via_cfg.hidden_coords, direct);
        assert!(
            !via_cfg.hidden_coords.is_empty(),
            "varying CPPN should yield a non-empty discovered hidden layer"
        );
    }

    // ── 27: validate catches bad quadtree params ──────────────────────────────

    #[test]
    fn validate_bad_quadtree_params() {
        let mut cfg = default_es_cfg();
        cfg.max_depth = 0;
        assert!(cfg.validate().is_err());

        let mut cfg = default_es_cfg();
        cfg.initial_depth = cfg.max_depth + 1;
        assert!(cfg.validate().is_err());

        let mut cfg = default_es_cfg();
        cfg.division_threshold = -1.0;
        assert!(cfg.validate().is_err());

        let mut cfg = default_es_cfg();
        cfg.band_threshold = -0.5;
        assert!(cfg.validate().is_err());
    }
}
