//! Adaptive quadtree cell hierarchy for 2D mesh refinement (AMR).
//!
//! A *quadtree* recursively subdivides a square (or rectangular) root domain
//! into four congruent children. This is the 2D analogue of an octree; the same
//! data structure and algorithms generalise directly to 3D by using eight
//! children. We implement the 2D variant for clarity, including:
//!
//! * [`Quadtree::refine`] — split a leaf into four children;
//! * [`Quadtree::coarsen`] — merge a parent's four leaf children back into a leaf;
//! * leaf iteration and per-cell level tracking;
//! * [`Quadtree::balance_2to1`] — enforce the **2:1 balance condition** (no two
//!   face-adjacent leaves differ by more than one refinement level), the standard
//!   constraint underpinning conforming AMR flux operators.
//!
//! # Storage
//!
//! Cells are stored in a flat arena (`Vec<Cell>`) indexed by `usize`; the root is
//! index `0`. Each cell records its level, its axis-aligned bounding box, an
//! optional parent, and (when refined) the indices of its four children. Children
//! are laid out in Morton (Z-order) quadrant order:
//!
//! ```text
//!   2 │ 3        (y increases upward)
//!  ───┼───
//!   0 │ 1        (x increases rightward)
//! ```

use crate::error::{PdeError, PdeResult};

/// The four child quadrants of a quadtree cell, in Z-order.
pub const CHILDREN_PER_CELL: usize = 4;

/// Axis-aligned 2D bounding box `[x_min, x_max] × [y_min, y_max]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl Aabb {
    /// Centre point `(cx, cy)`.
    #[must_use]
    pub fn center(&self) -> (f64, f64) {
        (
            0.5 * (self.x_min + self.x_max),
            0.5 * (self.y_min + self.y_max),
        )
    }

    /// Width (x-extent).
    #[must_use]
    pub fn width(&self) -> f64 {
        self.x_max - self.x_min
    }

    /// Height (y-extent).
    #[must_use]
    pub fn height(&self) -> f64 {
        self.y_max - self.y_min
    }

    /// Mean side length `(width + height) / 2`, used as the cell size `h`.
    #[must_use]
    pub fn size(&self) -> f64 {
        0.5 * (self.width() + self.height())
    }

    /// The four child boxes in Z-order (0:SW, 1:SE, 2:NW, 3:NE).
    fn quadrants(&self) -> [Aabb; CHILDREN_PER_CELL] {
        let (cx, cy) = self.center();
        [
            Aabb {
                x_min: self.x_min,
                y_min: self.y_min,
                x_max: cx,
                y_max: cy,
            },
            Aabb {
                x_min: cx,
                y_min: self.y_min,
                x_max: self.x_max,
                y_max: cy,
            },
            Aabb {
                x_min: self.x_min,
                y_min: cy,
                x_max: cx,
                y_max: self.y_max,
            },
            Aabb {
                x_min: cx,
                y_min: cy,
                x_max: self.x_max,
                y_max: self.y_max,
            },
        ]
    }
}

/// A single quadtree cell in the arena.
#[derive(Debug, Clone)]
pub struct Cell {
    /// Refinement level (root = 0).
    pub level: usize,
    /// Geometric extent.
    pub bbox: Aabb,
    /// Parent index, or `None` for the root.
    pub parent: Option<usize>,
    /// Child indices (length `4`) when refined; empty when a leaf.
    pub children: Vec<usize>,
}

impl Cell {
    /// True if the cell has no children.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// An adaptive 2D quadtree over a rectangular root domain.
#[derive(Debug, Clone)]
pub struct Quadtree {
    cells: Vec<Cell>,
}

impl Quadtree {
    /// Create a quadtree whose root spans `[x_min, x_max] × [y_min, y_max]`.
    ///
    /// # Errors
    /// [`PdeError::InvalidGrid`] if the domain is degenerate
    /// (`x_max <= x_min` or `y_max <= y_min`).
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> PdeResult<Self> {
        if x_max <= x_min || y_max <= y_min {
            return Err(PdeError::InvalidGrid(
                "quadtree: root domain must have positive extent".into(),
            ));
        }
        let root = Cell {
            level: 0,
            bbox: Aabb {
                x_min,
                y_min,
                x_max,
                y_max,
            },
            parent: None,
            children: Vec::new(),
        };
        Ok(Self { cells: vec![root] })
    }

    /// Number of cells in the arena (leaves + internal).
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Indices of all current leaf cells.
    #[must_use]
    pub fn leaves(&self) -> Vec<usize> {
        (0..self.cells.len())
            .filter(|&i| self.cells[i].is_leaf())
            .collect()
    }

    /// Number of leaf cells (the active mesh size).
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.cells.iter().filter(|c| c.is_leaf()).count()
    }

    /// Immutable access to a cell.
    ///
    /// # Errors
    /// [`PdeError::IndexOutOfBounds`] if `idx` is out of range.
    pub fn cell(&self, idx: usize) -> PdeResult<&Cell> {
        self.cells.get(idx).ok_or(PdeError::IndexOutOfBounds {
            index: idx,
            len: self.cells.len(),
        })
    }

    /// Maximum refinement level present among all cells.
    #[must_use]
    pub fn max_level(&self) -> usize {
        self.cells.iter().map(|c| c.level).max().unwrap_or(0)
    }

    /// Refine leaf `idx` into four children, returning the new child indices.
    ///
    /// # Errors
    /// * [`PdeError::IndexOutOfBounds`] if `idx` is out of range.
    /// * [`PdeError::InvalidParameter`] if `idx` is not currently a leaf.
    pub fn refine(&mut self, idx: usize) -> PdeResult<Vec<usize>> {
        if idx >= self.cells.len() {
            return Err(PdeError::IndexOutOfBounds {
                index: idx,
                len: self.cells.len(),
            });
        }
        if !self.cells[idx].is_leaf() {
            return Err(PdeError::InvalidParameter {
                name: "idx".into(),
                reason: "cannot refine a non-leaf cell".into(),
            });
        }
        let level = self.cells[idx].level;
        let quads = self.cells[idx].bbox.quadrants();
        let mut child_ids = Vec::with_capacity(CHILDREN_PER_CELL);
        for q in quads {
            let new_id = self.cells.len();
            self.cells.push(Cell {
                level: level + 1,
                bbox: q,
                parent: Some(idx),
                children: Vec::new(),
            });
            child_ids.push(new_id);
        }
        self.cells[idx].children = child_ids.clone();
        Ok(child_ids)
    }

    /// Coarsen cell `idx` by removing its four (leaf) children, turning it back
    /// into a leaf.
    ///
    /// To preserve arena index stability we do **not** physically delete the
    /// child slots; instead we mark them detached (parent cleared, level set so
    /// they are no longer reachable as leaves). Reachability for leaf iteration
    /// is by `Quadtree::is_reachable_leaf`; [`Quadtree::leaves`] and
    /// [`Quadtree::leaf_count`] both honour it via a compaction-free traversal.
    ///
    /// To keep iteration simple and correct, this method performs a full arena
    /// compaction: it rebuilds the tree omitting the detached children. Indices
    /// returned by earlier calls are therefore invalidated after a coarsen.
    ///
    /// # Errors
    /// * [`PdeError::IndexOutOfBounds`] if `idx` is out of range.
    /// * [`PdeError::InvalidParameter`] if `idx` is a leaf or any child is not
    ///   itself a leaf (cannot coarsen across more than one level at once).
    pub fn coarsen(&mut self, idx: usize) -> PdeResult<()> {
        if idx >= self.cells.len() {
            return Err(PdeError::IndexOutOfBounds {
                index: idx,
                len: self.cells.len(),
            });
        }
        if self.cells[idx].is_leaf() {
            return Err(PdeError::InvalidParameter {
                name: "idx".into(),
                reason: "cannot coarsen a leaf cell".into(),
            });
        }
        let children = self.cells[idx].children.clone();
        for &c in &children {
            if !self.cells[c].is_leaf() {
                return Err(PdeError::InvalidParameter {
                    name: "idx".into(),
                    reason: "all children must be leaves to coarsen".into(),
                });
            }
        }
        // Detach: clear children list; mark child cells as removed.
        self.cells[idx].children.clear();
        // Rebuild arena to drop the detached descendants and keep indices dense.
        self.compact();
        Ok(())
    }

    /// Rebuild the arena via a breadth-first re-walk from the root so that only
    /// reachable cells remain, with contiguous indices.
    fn compact(&mut self) {
        let old = std::mem::take(&mut self.cells);
        if old.is_empty() {
            return;
        }
        let mut new_cells: Vec<Cell> = Vec::with_capacity(old.len());
        // Map old index -> new index via a stack-based pre-order walk from root 0.
        let mut stack = vec![(0usize, None::<usize>)];
        while let Some((old_idx, new_parent)) = stack.pop() {
            let src = &old[old_idx];
            let new_idx = new_cells.len();
            new_cells.push(Cell {
                level: src.level,
                bbox: src.bbox,
                parent: new_parent,
                children: Vec::new(),
            });
            // Push children (reverse so Z-order is preserved on pop).
            for &c in src.children.iter().rev() {
                stack.push((c, Some(new_idx)));
            }
        }
        // Second pass: reconnect children by re-walking with the same order.
        // Because pre-order assigns parent before children, we can reconstruct
        // child lists by scanning new_cells for matching parents in order.
        let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); new_cells.len()];
        for (i, c) in new_cells.iter().enumerate() {
            if let Some(p) = c.parent {
                children_of[p].push(i);
            }
        }
        for (i, kids) in children_of.into_iter().enumerate() {
            new_cells[i].children = kids;
        }
        self.cells = new_cells;
    }

    /// Find face-adjacent leaf neighbours of leaf `idx` in the four cardinal
    /// directions. Returns the indices of leaves whose boxes share a (possibly
    /// partial) edge with cell `idx`.
    ///
    /// Two leaves are face-neighbours iff their boxes touch along a vertical or
    /// horizontal segment of positive length (a shared edge), not merely a
    /// corner.
    #[must_use]
    pub fn face_neighbors(&self, idx: usize) -> Vec<usize> {
        let mut out = Vec::new();
        if idx >= self.cells.len() || !self.cells[idx].is_leaf() {
            return out;
        }
        let a = self.cells[idx].bbox;
        for (j, cell) in self.cells.iter().enumerate() {
            if j == idx || !cell.is_leaf() {
                continue;
            }
            let b = cell.bbox;
            if Self::share_face(&a, &b) {
                out.push(j);
            }
        }
        out
    }

    /// True if boxes `a` and `b` share an edge (face) of positive length.
    fn share_face(a: &Aabb, b: &Aabb) -> bool {
        let eps = 1.0e-12 * (a.size() + b.size()).max(1.0);
        // Vertical shared edge: a.x_max == b.x_min or a.x_min == b.x_max, with
        // overlapping y-extent.
        let touch_x = (a.x_max - b.x_min).abs() < eps || (a.x_min - b.x_max).abs() < eps;
        let touch_y = (a.y_max - b.y_min).abs() < eps || (a.y_min - b.y_max).abs() < eps;
        let overlap_y = (a.y_min.max(b.y_min)) < (a.y_max.min(b.y_max)) - eps;
        let overlap_x = (a.x_min.max(b.x_min)) < (a.x_max.min(b.x_max)) - eps;
        (touch_x && overlap_y) || (touch_y && overlap_x)
    }

    /// Maximum absolute level difference across any pair of face-adjacent leaves.
    ///
    /// A value `≤ 1` certifies the 2:1 balance condition holds.
    #[must_use]
    pub fn max_neighbor_level_diff(&self) -> usize {
        let leaves = self.leaves();
        let mut worst = 0usize;
        for &l in &leaves {
            let ll = self.cells[l].level;
            for n in self.face_neighbors(l) {
                let diff = ll.abs_diff(self.cells[n].level);
                if diff > worst {
                    worst = diff;
                }
            }
        }
        worst
    }

    /// Enforce the **2:1 balance condition**: repeatedly refine any leaf that is
    /// more than one level coarser than a face neighbour, until no violations
    /// remain. Returns the number of refinement operations performed.
    ///
    /// The pass terminates because each refinement strictly increases the total
    /// number of cells while the maximum level is bounded by the deepest
    /// pre-existing cell (refinement only propagates coarser cells *up* toward,
    /// never beyond, the finest neighbour).
    ///
    /// # Errors
    /// Propagates [`PdeError`] from [`Quadtree::refine`] (should not occur for
    /// in-range leaves).
    pub fn balance_2to1(&mut self) -> PdeResult<usize> {
        let mut ops = 0usize;
        // Iterate to a fixed point. Each sweep collects violating leaves, refines
        // them, then re-scans (refinement can create new violations one ring out).
        loop {
            let leaves = self.leaves();
            let mut to_refine: Vec<usize> = Vec::new();
            for &l in &leaves {
                let ll = self.cells[l].level;
                for n in self.face_neighbors(l) {
                    let nl = self.cells[n].level;
                    if nl > ll + 1 {
                        to_refine.push(l);
                        break;
                    }
                }
            }
            if to_refine.is_empty() {
                break;
            }
            // Refine the collected leaves. Refining changes indices only by
            // appending, so previously collected (still-leaf) indices remain valid.
            for l in to_refine {
                if self.cells[l].is_leaf() {
                    self.refine(l)?;
                    ops += 1;
                }
            }
        }
        Ok(ops)
    }

    /// Sizes (mean side length) of all current leaves, in leaf-iteration order.
    #[must_use]
    pub fn leaf_sizes(&self) -> Vec<f64> {
        self.leaves()
            .into_iter()
            .map(|i| self.cells[i].bbox.size())
            .collect()
    }

    /// Centres of all current leaves, in leaf-iteration order.
    #[must_use]
    pub fn leaf_centers(&self) -> Vec<(f64, f64)> {
        self.leaves()
            .into_iter()
            .map(|i| self.cells[i].bbox.center())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_tree() -> Quadtree {
        Quadtree::new(0.0, 0.0, 1.0, 1.0).expect("root")
    }

    #[test]
    fn root_is_single_leaf() {
        let t = unit_tree();
        assert_eq!(t.leaf_count(), 1);
        assert_eq!(t.cell_count(), 1);
        assert_eq!(t.max_level(), 0);
    }

    #[test]
    fn refine_root_makes_four_children() {
        let mut t = unit_tree();
        let kids = t.refine(0).expect("refine");
        assert_eq!(kids.len(), 4);
        assert_eq!(t.leaf_count(), 4);
        assert_eq!(t.cell_count(), 5); // root + 4
        // Children cover the domain and have half size.
        for &k in &kids {
            let c = t.cell(k).expect("cell");
            assert_eq!(c.level, 1);
            assert!((c.bbox.size() - 0.5).abs() < 1e-12);
        }
        // Quadrant centres in Z-order.
        let centers: Vec<(f64, f64)> = kids
            .iter()
            .map(|&k| t.cell(k).expect("c").bbox.center())
            .collect();
        assert!((centers[0].0 - 0.25).abs() < 1e-12 && (centers[0].1 - 0.25).abs() < 1e-12);
        assert!((centers[3].0 - 0.75).abs() < 1e-12 && (centers[3].1 - 0.75).abs() < 1e-12);
    }

    #[test]
    fn refine_then_coarsen_round_trip() {
        let mut t = unit_tree();
        t.refine(0).expect("refine");
        assert_eq!(t.leaf_count(), 4);
        // After coarsening the root, we are back to a single leaf.
        t.coarsen(0).expect("coarsen");
        assert_eq!(t.leaf_count(), 1);
        assert_eq!(t.max_level(), 0);
        // The single remaining leaf is the full domain.
        let leaves = t.leaves();
        assert_eq!(leaves.len(), 1);
        let b = t.cell(leaves[0]).expect("c").bbox;
        assert!((b.size() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn multilevel_refine_counts() {
        let mut t = unit_tree();
        let kids = t.refine(0).expect("r");
        // Refine the south-west child once more.
        t.refine(kids[0]).expect("r2");
        // Leaves: 3 of the original children + 4 grandchildren = 7.
        assert_eq!(t.leaf_count(), 7);
        assert_eq!(t.max_level(), 2);
    }

    #[test]
    fn face_neighbors_of_quadrants() {
        let mut t = unit_tree();
        let kids = t.refine(0).expect("r");
        // SW (0) shares faces with SE (1) and NW (2), shares only a corner with NE (3).
        let nbrs = t.face_neighbors(kids[0]);
        assert!(nbrs.contains(&kids[1]), "SW must neighbour SE");
        assert!(nbrs.contains(&kids[2]), "SW must neighbour NW");
        assert!(
            !nbrs.contains(&kids[3]),
            "SW must NOT face-neighbour NE (corner)"
        );
    }

    #[test]
    fn balance_2to1_enforced_after_deep_refine() {
        let mut t = unit_tree();
        let l1 = t.refine(0).expect("r"); // level-1 children of the root
        // Refine the SW child, then refine *its* NE grandchild ([0.25,0.5]²).
        // That grandchild's level-3 children touch the level-1 SE/NW cells at
        // x = 0.5 / y = 0.5, creating a level-3 ↔ level-1 (diff 2) violation.
        let l2 = t.refine(l1[0]).expect("r");
        let _l3 = t.refine(l2[3]).expect("r");
        assert!(
            t.max_neighbor_level_diff() >= 2,
            "setup should be unbalanced, got diff {}",
            t.max_neighbor_level_diff()
        );
        let ops = t.balance_2to1().expect("balance");
        assert!(ops > 0, "balance must perform refinements");
        // After balancing, no face-adjacent leaves differ by more than one level.
        assert!(
            t.max_neighbor_level_diff() <= 1,
            "2:1 balance violated: {}",
            t.max_neighbor_level_diff()
        );
    }

    #[test]
    fn balance_idempotent_on_balanced_tree() {
        let mut t = unit_tree();
        t.refine(0).expect("r");
        // Uniformly refined tree is already balanced.
        assert!(t.max_neighbor_level_diff() <= 1);
        let ops = t.balance_2to1().expect("balance");
        assert_eq!(ops, 0, "already-balanced tree needs no refinement");
    }

    #[test]
    fn cannot_refine_non_leaf() {
        let mut t = unit_tree();
        t.refine(0).expect("r");
        assert!(t.refine(0).is_err());
    }

    #[test]
    fn coarsen_requires_leaf_children() {
        let mut t = unit_tree();
        let kids = t.refine(0).expect("r");
        t.refine(kids[0]).expect("r2");
        // Root's child 0 is no longer a leaf, so coarsening the root is invalid.
        assert!(t.coarsen(0).is_err());
    }
}
