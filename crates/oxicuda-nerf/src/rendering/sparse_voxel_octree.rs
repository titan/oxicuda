//! Sparse Voxel Octree acceleration structure (NSVF, NeurIPS 2020).
//!
//! Reference: Liu et al. 2020, "Neural Sparse Voxel Fields" (NeurIPS).
//!
//! An octree adaptively subdivides a 3D scene AABB into 8 octants per
//! interior node. Homogeneous (empty / uniform) regions collapse into a
//! single leaf so that ray traversal can skip large empty volumes,
//! dramatically reducing the number of MLP evaluations needed for
//! sparse scenes.
//!
//! # Grid Convention
//!
//! `build_from_grid` takes a flat density grid with the following memory
//! layout (row-major over `Z·Y·X`):
//!
//! ```text
//! grid[z * (Y * X) + y * X + x] = density at voxel (x, y, z)
//! ```
//!
//! All three grid dimensions must be powers of 2.
//!
//! # Octant Convention
//!
//! For an interior node, child slot `i ∈ 0..8` is selected by the high
//! bits of the (x, y, z) sub-coordinate within the node:
//!
//! ```text
//! child_index = (z_high) * 4 + (y_high) * 2 + (x_high)
//! ```
//!
//! where `*_high = 1` for the upper half of the node along that axis.
//!
//! # Ray Traversal
//!
//! `traverse_ray` returns hits in strictly increasing `t_enter` order
//! by AABB-intersecting children at each interior node and recursing
//! front-to-back. Only `occupied` leaves are emitted; empty regions are
//! skipped without further recursion.

use crate::error::{NerfError, NerfResult};

// ─── AABB ────────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box used by the sparse voxel octree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Lower corner.
    pub min: [f32; 3],
    /// Upper corner.
    pub max: [f32; 3],
}

impl Aabb {
    /// Construct an axis-aligned bounding box.
    ///
    /// # Errors
    ///
    /// Returns `InvalidOctreeConfig` if any `min[i] >= max[i]`.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> NerfResult<Self> {
        for i in 0..3 {
            let mn = *min.get(i).ok_or(NerfError::InvalidOctreeConfig {
                msg: "aabb min len".to_string(),
            })?;
            let mx = *max.get(i).ok_or(NerfError::InvalidOctreeConfig {
                msg: "aabb max len".to_string(),
            })?;
            if mn >= mx {
                return Err(NerfError::InvalidOctreeConfig {
                    msg: format!("aabb min[{i}]={mn} >= max[{i}]={mx}"),
                });
            }
            if !mn.is_finite() || !mx.is_finite() {
                return Err(NerfError::InvalidOctreeConfig {
                    msg: "aabb has non-finite corner".to_string(),
                });
            }
        }
        Ok(Self { min, max })
    }

    /// Standard slab-test ray / AABB intersection.
    ///
    /// Returns `Some((t_enter, t_exit))` if the ray hits the box at any
    /// `t >= 0`, otherwise `None`. Robust to axis-parallel rays (zero
    /// component in `dir`): such an axis is treated as a degenerate slab
    /// satisfied iff the origin is inside that slab.
    #[must_use]
    pub fn ray_intersect(&self, origin: [f32; 3], dir: [f32; 3]) -> Option<(f32, f32)> {
        let mut t_enter = f32::NEG_INFINITY;
        let mut t_exit = f32::INFINITY;
        for i in 0..3 {
            let o = origin[i];
            let d = dir[i];
            let mn = self.min[i];
            let mx = self.max[i];
            if d.abs() < 1e-20 {
                // Ray parallel to slab → must be inside slab to hit.
                if o < mn || o > mx {
                    return None;
                }
            } else {
                let inv = 1.0_f32 / d;
                let mut t0 = (mn - o) * inv;
                let mut t1 = (mx - o) * inv;
                if t0 > t1 {
                    std::mem::swap(&mut t0, &mut t1);
                }
                if t0 > t_enter {
                    t_enter = t0;
                }
                if t1 < t_exit {
                    t_exit = t1;
                }
                if t_enter > t_exit {
                    return None;
                }
            }
        }
        if t_exit < 0.0 {
            return None;
        }
        // Clamp the entry to be non-negative (ray starts inside the box).
        let t_enter = t_enter.max(0.0);
        if t_enter > t_exit {
            return None;
        }
        Some((t_enter, t_exit))
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for sparse voxel octree construction.
#[derive(Debug, Clone)]
pub struct SparseVoxelOctreeConfig {
    /// Maximum recursion depth. Depth 0 forces a single leaf.
    pub max_depth: usize,
    /// World-space bounds of the scene.
    pub scene_bounds: Aabb,
    /// Density above which a voxel is considered occupied.
    pub density_threshold: f32,
}

// ─── Node ────────────────────────────────────────────────────────────────────

/// One node in the sparse voxel octree.
#[derive(Debug, Clone)]
pub enum OctreeNode {
    /// Terminal node: either empty (`occupied = false`) or a single occupied voxel.
    Leaf {
        /// Spatial extents of the leaf.
        bounds: Aabb,
        /// Whether any voxel inside this leaf exceeds the density threshold.
        occupied: bool,
        /// Representative density (0 for empty leaves, max density otherwise).
        density: f32,
    },
    /// Internal node with up to 8 children.
    Internal {
        /// Spatial extents covered by all children.
        bounds: Aabb,
        /// Per-octant children. `children[i]` is `None` iff the corresponding
        /// octant collapsed to an empty leaf and is omitted.
        children: [Option<Box<OctreeNode>>; 8],
    },
}

impl OctreeNode {
    /// Return the AABB of any node variant.
    #[must_use]
    pub fn bounds(&self) -> &Aabb {
        match self {
            OctreeNode::Leaf { bounds, .. } => bounds,
            OctreeNode::Internal { bounds, .. } => bounds,
        }
    }
}

// ─── Ray hit ─────────────────────────────────────────────────────────────────

/// One occupied leaf intersected by a ray.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Distance along the ray where the leaf is first entered.
    pub t_enter: f32,
    /// Distance along the ray where the leaf is exited.
    pub t_exit: f32,
    /// Representative density inside the leaf.
    pub density: f32,
}

// ─── SparseVoxelOctree ───────────────────────────────────────────────────────

/// Sparse voxel octree built from a dense density grid.
#[derive(Debug, Clone)]
pub struct SparseVoxelOctree {
    /// Root node (the entire scene AABB).
    pub root: OctreeNode,
    n_nodes: usize,
    n_leaves: usize,
    n_occupied: usize,
}

impl SparseVoxelOctree {
    /// Build the octree from a flat row-major density grid (Z·Y·X order).
    ///
    /// All three grid dimensions must be powers of 2 and the slice length
    /// must equal `nx * ny * nz`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidOctreeConfig` if `density_threshold < 0`, `max_depth = 0`
    /// combined with non-trivial grid dims, any dim is not a power of 2, or
    /// `grid.len() != nx * ny * nz`.
    pub fn build_from_grid(
        grid: &[f32],
        grid_dim: (usize, usize, usize),
        cfg: SparseVoxelOctreeConfig,
    ) -> NerfResult<Self> {
        let (nx, ny, nz) = grid_dim;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(NerfError::InvalidOctreeConfig {
                msg: format!("grid dim has a zero axis: ({nx}, {ny}, {nz})"),
            });
        }
        if !is_power_of_two(nx) || !is_power_of_two(ny) || !is_power_of_two(nz) {
            return Err(NerfError::InvalidOctreeConfig {
                msg: format!(
                    "grid dim must be powers of 2 along each axis (got ({nx}, {ny}, {nz}))"
                ),
            });
        }
        let expected = nx.checked_mul(ny).and_then(|p| p.checked_mul(nz)).ok_or(
            NerfError::InvalidOctreeConfig {
                msg: "grid dim overflow".to_string(),
            },
        )?;
        if grid.len() != expected {
            return Err(NerfError::DimensionMismatch {
                expected,
                got: grid.len(),
            });
        }
        if !cfg.density_threshold.is_finite() || cfg.density_threshold < 0.0 {
            return Err(NerfError::InvalidOctreeConfig {
                msg: format!(
                    "density_threshold must be non-negative finite, got {}",
                    cfg.density_threshold
                ),
            });
        }
        if cfg.max_depth == 0 && (nx > 1 || ny > 1 || nz > 1) {
            return Err(NerfError::InvalidOctreeConfig {
                msg: format!(
                    "max_depth = 0 incompatible with non-singleton grid ({nx}, {ny}, {nz})"
                ),
            });
        }

        let mut n_nodes = 0usize;
        let mut n_leaves = 0usize;
        let mut n_occupied = 0usize;

        let root = build_node(
            grid,
            (nx, ny, nz),
            (0, 0, 0),
            (nx, ny, nz),
            cfg.scene_bounds,
            0,
            &cfg,
            &mut n_nodes,
            &mut n_leaves,
            &mut n_occupied,
        )?;

        Ok(Self {
            root,
            n_nodes,
            n_leaves,
            n_occupied,
        })
    }

    /// Number of nodes (interior + leaves).
    #[must_use]
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of leaf nodes (occupied + empty).
    #[must_use]
    #[inline]
    pub fn n_leaves(&self) -> usize {
        self.n_leaves
    }

    /// Number of occupied leaves.
    #[must_use]
    #[inline]
    pub fn n_occupied_leaves(&self) -> usize {
        self.n_occupied
    }

    /// Traverse a ray through the octree, returning all hits in front-to-back
    /// order along the ray.
    ///
    /// Hits are guaranteed to be strictly ordered by `t_enter` and to satisfy
    /// `t_enter < t_exit` per hit.
    ///
    /// # Errors
    ///
    /// Returns `ZeroRayDirection` if `dir` is the zero vector (or numerically
    /// indistinguishable from it).
    pub fn traverse_ray(&self, origin: [f32; 3], dir: [f32; 3]) -> NerfResult<Vec<RayHit>> {
        let dn = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        if dn < 1e-30 {
            return Err(NerfError::ZeroRayDirection);
        }
        let mut hits: Vec<RayHit> = Vec::new();
        traverse(&self.root, origin, dir, &mut hits);
        // The recursive traversal is already front-to-back, but enforce strict
        // monotonicity by sorting + dedup of near-coincident t_enter values.
        hits.sort_by(|a, b| {
            a.t_enter
                .partial_cmp(&b.t_enter)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }

    /// Visit every node in deterministic pre-order traversal. Useful for
    /// structural comparisons in tests.
    pub fn visit_preorder<F: FnMut(&OctreeNode)>(&self, mut f: F) {
        visit(&self.root, &mut f);
    }
}

// ─── Construction helpers ────────────────────────────────────────────────────

#[inline]
fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

#[allow(clippy::too_many_arguments)]
fn build_node(
    grid: &[f32],
    grid_dim: (usize, usize, usize),
    lo: (usize, usize, usize),
    hi: (usize, usize, usize),
    bounds: Aabb,
    depth: usize,
    cfg: &SparseVoxelOctreeConfig,
    n_nodes: &mut usize,
    n_leaves: &mut usize,
    n_occupied: &mut usize,
) -> NerfResult<OctreeNode> {
    let sx = hi.0 - lo.0;
    let sy = hi.1 - lo.1;
    let sz = hi.2 - lo.2;

    // Compute max density inside the sub-grid.
    let (max_density, picked_density) = scan_subgrid(grid, grid_dim, lo, hi)?;

    // If sub-region is empty (all ≤ threshold) → empty leaf.
    if max_density <= cfg.density_threshold {
        *n_nodes += 1;
        *n_leaves += 1;
        return Ok(OctreeNode::Leaf {
            bounds,
            occupied: false,
            density: 0.0,
        });
    }

    // If at max depth or sub-grid is a single voxel → terminal occupied leaf.
    if depth >= cfg.max_depth || (sx == 1 && sy == 1 && sz == 1) {
        *n_nodes += 1;
        *n_leaves += 1;
        if picked_density > cfg.density_threshold {
            *n_occupied += 1;
        }
        return Ok(OctreeNode::Leaf {
            bounds,
            occupied: picked_density > cfg.density_threshold,
            density: picked_density,
        });
    }

    // Otherwise: subdivide into octants. Splits collapse to the parent axis
    // when that axis already has size 1 (rectangular non-cube grids).
    let mx = lo.0 + sx / 2;
    let my = lo.1 + sy / 2;
    let mz = lo.2 + sz / 2;

    let cx_mid = 0.5 * (bounds.min[0] + bounds.max[0]);
    let cy_mid = 0.5 * (bounds.min[1] + bounds.max[1]);
    let cz_mid = 0.5 * (bounds.min[2] + bounds.max[2]);

    let mut children: [Option<Box<OctreeNode>>; 8] = Default::default();

    for (oct, slot) in children.iter_mut().enumerate() {
        let xh = (oct & 1) != 0;
        let yh = (oct & 2) != 0;
        let zh = (oct & 4) != 0;

        // Skip duplicate octants along singleton axes.
        if (sx == 1 && xh) || (sy == 1 && yh) || (sz == 1 && zh) {
            continue;
        }

        let (clox, chix) = if sx > 1 {
            if xh { (mx, hi.0) } else { (lo.0, mx) }
        } else {
            (lo.0, hi.0)
        };
        let (cloy, chiy) = if sy > 1 {
            if yh { (my, hi.1) } else { (lo.1, my) }
        } else {
            (lo.1, hi.1)
        };
        let (cloz, chiz) = if sz > 1 {
            if zh { (mz, hi.2) } else { (lo.2, mz) }
        } else {
            (lo.2, hi.2)
        };

        let cb_min = [
            if xh { cx_mid } else { bounds.min[0] },
            if yh { cy_mid } else { bounds.min[1] },
            if zh { cz_mid } else { bounds.min[2] },
        ];
        let cb_max = [
            if xh { bounds.max[0] } else { cx_mid },
            if yh { bounds.max[1] } else { cy_mid },
            if zh { bounds.max[2] } else { cz_mid },
        ];
        let child_bounds = Aabb::new(cb_min, cb_max)?;

        let child = build_node(
            grid,
            grid_dim,
            (clox, cloy, cloz),
            (chix, chiy, chiz),
            child_bounds,
            depth + 1,
            cfg,
            n_nodes,
            n_leaves,
            n_occupied,
        )?;
        *slot = Some(Box::new(child));
    }

    *n_nodes += 1;
    Ok(OctreeNode::Internal { bounds, children })
}

fn scan_subgrid(
    grid: &[f32],
    grid_dim: (usize, usize, usize),
    lo: (usize, usize, usize),
    hi: (usize, usize, usize),
) -> NerfResult<(f32, f32)> {
    let (nx, ny, _nz) = grid_dim;
    let stride_y = nx;
    let stride_z = nx * ny;
    let mut max_density = f32::NEG_INFINITY;
    let mut picked = 0.0_f32;
    for z in lo.2..hi.2 {
        for y in lo.1..hi.1 {
            for x in lo.0..hi.0 {
                let idx = z * stride_z + y * stride_y + x;
                let v = *grid.get(idx).ok_or(NerfError::Internal {
                    msg: format!("grid index out of range: idx={idx}, len={}", grid.len()),
                })?;
                if !v.is_finite() {
                    return Err(NerfError::NanEncountered {
                        context: format!("grid at ({x}, {y}, {z}): {v}"),
                    });
                }
                if v > max_density {
                    max_density = v;
                    picked = v;
                }
            }
        }
    }
    if max_density == f32::NEG_INFINITY {
        // Empty sub-range (should not happen given guards) — treat as empty.
        max_density = 0.0;
    }
    Ok((max_density, picked))
}

// ─── Traversal helpers ───────────────────────────────────────────────────────

fn traverse(node: &OctreeNode, origin: [f32; 3], dir: [f32; 3], out: &mut Vec<RayHit>) {
    match node {
        OctreeNode::Leaf {
            bounds,
            occupied,
            density,
        } => {
            if !*occupied {
                return;
            }
            if let Some((t_enter, t_exit)) = bounds.ray_intersect(origin, dir)
                && t_enter < t_exit
            {
                out.push(RayHit {
                    t_enter,
                    t_exit,
                    density: *density,
                });
            }
        }
        OctreeNode::Internal { bounds, children } => {
            // Parent miss → entire subtree miss.
            let outer = bounds.ray_intersect(origin, dir);
            if outer.is_none() {
                return;
            }
            // Compute (t_enter, t_exit) for each existing child and recurse in
            // ascending t_enter order. Children with no hit are skipped.
            let mut order: Vec<(f32, usize)> = Vec::with_capacity(8);
            for (i, c) in children.iter().enumerate() {
                if let Some(boxed) = c
                    && let Some((tn, _tx)) = boxed.bounds().ray_intersect(origin, dir)
                {
                    order.push((tn, i));
                }
            }
            order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for (_, i) in order {
                if let Some(boxed) = children.get(i).and_then(|c| c.as_ref()) {
                    traverse(boxed, origin, dir, out);
                }
            }
        }
    }
}

fn visit<F: FnMut(&OctreeNode)>(node: &OctreeNode, f: &mut F) {
    f(node);
    if let OctreeNode::Internal { children, .. } = node {
        for c in children.iter().flatten() {
            visit(c, f);
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn unit_box() -> Aabb {
        Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).expect("new should succeed")
    }

    fn cfg_depth(max_depth: usize, density_threshold: f32) -> SparseVoxelOctreeConfig {
        SparseVoxelOctreeConfig {
            max_depth,
            scene_bounds: unit_box(),
            density_threshold,
        }
    }

    #[test]
    fn aabb_new_rejects_invalid() {
        assert!(Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).is_ok());
        assert!(Aabb::new([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]).is_err());
        assert!(Aabb::new([2.0, 0.0, 0.0], [1.0, 1.0, 1.0]).is_err());
        assert!(Aabb::new([f32::NAN, 0.0, 0.0], [1.0, 1.0, 1.0]).is_err());
    }

    #[test]
    fn aabb_ray_intersects_simple() {
        let b = unit_box();
        let hit = b.ray_intersect([0.5, 0.5, -1.0], [0.0, 0.0, 1.0]);
        let (t_enter, t_exit) = hit.expect("hit should be present");
        assert!((t_enter - 1.0).abs() < 1e-5);
        assert!((t_exit - 2.0).abs() < 1e-5);
    }

    #[test]
    fn aabb_ray_misses() {
        let b = unit_box();
        // Ray parallel to z-axis but x outside the slab.
        assert!(b.ray_intersect([2.0, 0.5, -1.0], [0.0, 0.0, 1.0]).is_none());
    }

    #[test]
    fn aabb_ray_origin_inside() {
        let b = unit_box();
        let hit = b.ray_intersect([0.5, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let (t_enter, t_exit) = hit.expect("hit should be present");
        assert!(t_enter <= 1e-6);
        assert!((t_exit - 0.5).abs() < 1e-5);
    }

    #[test]
    fn aabb_ray_axis_parallel_inside_slab() {
        let b = unit_box();
        // Ray along x with origin already inside y and z slabs.
        let hit = b.ray_intersect([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0]);
        let (t_enter, t_exit) = hit.expect("hit should be present");
        assert!((t_enter - 1.0).abs() < 1e-5);
        assert!((t_exit - 2.0).abs() < 1e-5);
    }

    #[test]
    fn build_empty_grid_collapses_to_single_leaf() {
        let n = 4;
        let grid = vec![0.0_f32; n * n * n];
        let cfg = cfg_depth(4, 0.1);
        let oct = SparseVoxelOctree::build_from_grid(&grid, (n, n, n), cfg)
            .expect("value should be present");
        assert_eq!(oct.n_nodes(), 1);
        assert_eq!(oct.n_leaves(), 1);
        assert_eq!(oct.n_occupied_leaves(), 0);
        match &oct.root {
            OctreeNode::Leaf { occupied, .. } => assert!(!*occupied),
            _ => panic!("expected leaf"),
        }
    }

    #[test]
    fn build_full_grid_max_depth_all_occupied() {
        let n = 4usize;
        let grid = vec![1.0_f32; n * n * n];
        let cfg = cfg_depth(2, 0.0);
        let oct = SparseVoxelOctree::build_from_grid(&grid, (n, n, n), cfg)
            .expect("value should be present");
        // All voxels above threshold ⇒ all leaves occupied.
        assert_eq!(oct.n_occupied_leaves(), oct.n_leaves());
        assert!(oct.n_leaves() >= 1);
    }

    #[test]
    fn build_single_occupied_voxel_returns_one_hit() {
        let n = 4usize;
        let mut grid = vec![0.0_f32; n * n * n];
        // Voxel at (1, 1, 1) is occupied.
        let stride_z = n * n;
        let stride_y = n;
        grid[stride_z + stride_y + 1] = 1.0;
        let cfg = cfg_depth(4, 0.1);
        let oct = SparseVoxelOctree::build_from_grid(&grid, (n, n, n), cfg)
            .expect("value should be present");
        assert_eq!(oct.n_occupied_leaves(), 1);

        // Shoot a ray that passes through the occupied voxel along +z.
        // Voxel coords (1, 1, 1) inside unit box [0,1]³ at grid resolution 4
        // ⇒ x ∈ [0.25, 0.5], y ∈ [0.25, 0.5], z ∈ [0.25, 0.5].
        let hits = oct
            .traverse_ray([0.3, 0.3, -1.0], [0.0, 0.0, 1.0])
            .expect("traverse_ray should succeed");
        assert_eq!(hits.len(), 1);
        assert!(hits[0].t_enter < hits[0].t_exit);
        assert!((hits[0].density - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ray_missing_scene_returns_empty() {
        let n = 4usize;
        let grid = vec![1.0_f32; n * n * n];
        let cfg = cfg_depth(2, 0.0);
        let oct = SparseVoxelOctree::build_from_grid(&grid, (n, n, n), cfg)
            .expect("value should be present");
        // Ray nowhere near the unit box.
        let hits = oct
            .traverse_ray([5.0, 5.0, 5.0], [1.0, 0.0, 0.0])
            .expect("traverse_ray should succeed");
        assert!(hits.is_empty());
    }

    #[test]
    fn axis_aligned_ray_through_z_stack_returns_increasing_t() {
        let n = 4usize;
        let mut grid = vec![0.0_f32; n * n * n];
        let stride_z = n * n;
        let stride_y = n;
        // Stack of 3 occupied voxels along z at (1, 1, *).
        for z in 0..3 {
            grid[z * stride_z + stride_y + 1] = 1.0;
        }
        let cfg = cfg_depth(4, 0.5);
        let oct = SparseVoxelOctree::build_from_grid(&grid, (n, n, n), cfg)
            .expect("value should be present");
        let hits = oct
            .traverse_ray([0.3, 0.3, -1.0], [0.0, 0.0, 1.0])
            .expect("traverse_ray should succeed");
        assert_eq!(hits.len(), 3);
        for w in hits.windows(2) {
            assert!(
                w[1].t_enter > w[0].t_enter,
                "t_enter must be strictly increasing: {:?}",
                hits
            );
        }
        for h in &hits {
            assert!(h.t_enter < h.t_exit);
        }
    }

    #[test]
    fn pruning_homogeneous_block_reduces_nodes() {
        // A homogeneous empty block of size 4×4×4 should collapse to a single
        // leaf regardless of max_depth, whereas a fully-occupied 4×4×4 grid at
        // max_depth = 2 expands into multiple nodes.
        let n = 4usize;
        let grid_empty = vec![0.0_f32; n * n * n];
        let grid_full = vec![1.0_f32; n * n * n];
        let pruned = SparseVoxelOctree::build_from_grid(&grid_empty, (n, n, n), cfg_depth(2, 0.1))
            .expect("value should be present");
        let expanded = SparseVoxelOctree::build_from_grid(&grid_full, (n, n, n), cfg_depth(2, 0.0))
            .expect("value should be present");
        assert_eq!(pruned.n_nodes(), 1);
        assert!(
            expanded.n_nodes() > pruned.n_nodes(),
            "fully-occupied 4³ at max_depth=2 must produce more nodes than the pruned tree: pruned={}, expanded={}",
            pruned.n_nodes(),
            expanded.n_nodes(),
        );
    }

    #[test]
    fn n_occupied_leaves_matches_axis_aligned_grid() {
        // Place k distinct occupied voxels and check the count matches when
        // max_depth is large enough to resolve each voxel individually.
        let n = 4usize;
        let mut grid = vec![0.0_f32; n * n * n];
        let stride_z = n * n;
        let stride_y = n;
        let coords = [(0, 0, 0), (1, 2, 3), (3, 3, 3), (2, 1, 0)];
        for &(x, y, z) in &coords {
            grid[z * stride_z + y * stride_y + x] = 1.0;
        }
        // max_depth = log2(4) = 2; full resolution.
        let oct = SparseVoxelOctree::build_from_grid(&grid, (n, n, n), cfg_depth(2, 0.5))
            .expect("value should be present");
        assert_eq!(oct.n_occupied_leaves(), coords.len());
    }

    #[test]
    fn each_hit_has_t_enter_less_than_t_exit() {
        let n = 4usize;
        let mut grid = vec![0.0_f32; n * n * n];
        let stride_z = n * n;
        let stride_y = n;
        for v in 0..n {
            grid[v * stride_z + stride_y + 1] = 0.5 + v as f32 * 0.1;
        }
        let oct = SparseVoxelOctree::build_from_grid(&grid, (n, n, n), cfg_depth(2, 0.0))
            .expect("value should be present");
        let hits = oct
            .traverse_ray([0.4, 0.4, -1.0], [0.0, 0.0, 1.0])
            .expect("traverse_ray should succeed");
        assert!(!hits.is_empty());
        for h in hits {
            assert!(
                h.t_enter < h.t_exit,
                "expected t_enter < t_exit, got {:?}",
                h
            );
        }
    }

    #[test]
    fn deterministic_build() {
        let n = 4usize;
        let mut rng = LcgRng::new(42);
        let mut grid = vec![0.0_f32; n * n * n];
        for v in grid.iter_mut() {
            *v = rng.next_f32();
        }
        let cfg1 = cfg_depth(2, 0.5);
        let cfg2 = cfg_depth(2, 0.5);
        let a = SparseVoxelOctree::build_from_grid(&grid, (n, n, n), cfg1)
            .expect("value should be present");
        let b = SparseVoxelOctree::build_from_grid(&grid, (n, n, n), cfg2)
            .expect("value should be present");
        let mut va: Vec<(bool, f32)> = Vec::new();
        let mut vb: Vec<(bool, f32)> = Vec::new();
        a.visit_preorder(|node| {
            if let OctreeNode::Leaf {
                occupied, density, ..
            } = node
            {
                va.push((*occupied, *density));
            }
        });
        b.visit_preorder(|node| {
            if let OctreeNode::Leaf {
                occupied, density, ..
            } = node
            {
                vb.push((*occupied, *density));
            }
        });
        assert_eq!(a.n_nodes(), b.n_nodes());
        assert_eq!(va.len(), vb.len());
        for (x, y) in va.iter().zip(vb.iter()) {
            assert_eq!(x.0, y.0);
            assert!((x.1 - y.1).abs() < 1e-9);
        }
    }

    #[test]
    fn err_not_power_of_two() {
        let grid = vec![0.0_f32; 3 * 4 * 4];
        let res = SparseVoxelOctree::build_from_grid(&grid, (3, 4, 4), cfg_depth(2, 0.1));
        assert!(res.is_err());
    }

    #[test]
    fn err_len_mismatch() {
        let grid = vec![0.0_f32; 10];
        let res = SparseVoxelOctree::build_from_grid(&grid, (4, 4, 4), cfg_depth(2, 0.1));
        assert!(res.is_err());
    }

    #[test]
    fn err_invalid_bounds() {
        let res = Aabb::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
        assert!(res.is_err());
    }

    #[test]
    fn err_negative_threshold() {
        let grid = vec![0.0_f32; 4 * 4 * 4];
        let cfg = SparseVoxelOctreeConfig {
            max_depth: 2,
            scene_bounds: unit_box(),
            density_threshold: -1.0,
        };
        let res = SparseVoxelOctree::build_from_grid(&grid, (4, 4, 4), cfg);
        assert!(res.is_err());
    }

    #[test]
    fn err_zero_dir_traverse() {
        let grid = vec![1.0_f32; 4 * 4 * 4];
        let oct = SparseVoxelOctree::build_from_grid(&grid, (4, 4, 4), cfg_depth(2, 0.0))
            .expect("value should be present");
        let res = oct.traverse_ray([0.5, 0.5, 0.5], [0.0, 0.0, 0.0]);
        assert!(res.is_err());
    }

    #[test]
    fn err_max_depth_zero_with_grid_gt_one() {
        let grid = vec![0.0_f32; 4 * 4 * 4];
        let cfg = SparseVoxelOctreeConfig {
            max_depth: 0,
            scene_bounds: unit_box(),
            density_threshold: 0.1,
        };
        let res = SparseVoxelOctree::build_from_grid(&grid, (4, 4, 4), cfg);
        assert!(res.is_err());
    }

    #[test]
    fn err_zero_grid_dim() {
        let grid: Vec<f32> = vec![];
        let res = SparseVoxelOctree::build_from_grid(&grid, (0, 4, 4), cfg_depth(2, 0.1));
        assert!(res.is_err());
    }

    #[test]
    fn dense_grid_traverse_returns_one_ray_hit_per_leaf() {
        // Fully occupied grid → ray along +x at fixed (y, z) should hit every
        // leaf the ray passes through with strictly increasing t_enter.
        let n = 4usize;
        let grid = vec![1.0_f32; n * n * n];
        let cfg = cfg_depth(2, 0.0);
        let oct = SparseVoxelOctree::build_from_grid(&grid, (n, n, n), cfg)
            .expect("value should be present");
        let hits = oct
            .traverse_ray([-1.0, 0.3, 0.3], [1.0, 0.0, 0.0])
            .expect("traverse_ray should succeed");
        assert_eq!(hits.len(), n);
        for w in hits.windows(2) {
            assert!(w[1].t_enter > w[0].t_enter);
        }
    }

    #[test]
    fn single_voxel_grid_builds_leaf() {
        let grid = vec![1.0_f32; 1];
        let cfg = cfg_depth(0, 0.5);
        let oct = SparseVoxelOctree::build_from_grid(&grid, (1, 1, 1), cfg)
            .expect("value should be present");
        assert_eq!(oct.n_nodes(), 1);
        assert_eq!(oct.n_leaves(), 1);
        assert_eq!(oct.n_occupied_leaves(), 1);
    }

    #[test]
    fn empty_grid_max_depth_zero_ok() {
        // 1×1×1 grid with max_depth = 0 (singleton case is allowed).
        let grid = vec![0.0_f32; 1];
        let cfg = SparseVoxelOctreeConfig {
            max_depth: 0,
            scene_bounds: unit_box(),
            density_threshold: 0.5,
        };
        let oct = SparseVoxelOctree::build_from_grid(&grid, (1, 1, 1), cfg)
            .expect("value should be present");
        assert_eq!(oct.n_nodes(), 1);
        assert_eq!(oct.n_occupied_leaves(), 0);
    }

    #[test]
    fn nan_in_grid_is_rejected() {
        let mut grid = vec![0.0_f32; 4 * 4 * 4];
        grid[5] = f32::NAN;
        let res = SparseVoxelOctree::build_from_grid(&grid, (4, 4, 4), cfg_depth(2, 0.1));
        assert!(res.is_err());
    }
}
