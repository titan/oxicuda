//! Robust Random Cut Forest (RRCF) — Guha, Mishra, Roy & Schrijvers, ICML 2016.
//!
//! A Robust Random Cut Tree (RRCT) is a randomized binary space-partitioning
//! tree built so that the cut at every internal node is chosen as follows:
//!
//! 1. Let `B = [ℓ₁,h₁] × … × [ℓ_d,h_d]` be the axis-aligned bounding box of the
//!    points currently under the node, and let `Lᵢ = hᵢ − ℓᵢ` be the side
//!    length in dimension `i`.
//! 2. Pick the cut **dimension** `i` with probability `Lᵢ / Σⱼ Lⱼ`.
//! 3. Pick the cut **value** `p` uniformly at random in `[ℓᵢ, hᵢ]`.
//! 4. Points with `xᵢ ≤ p` recurse to the left child, the rest to the right.
//!
//! Because heavy (long) dimensions are cut more often, the construction is
//! provably *robust* to irrelevant dimensions and scale.
//!
//! # Anomaly score — Collusive Displacement (CoDisp)
//!
//! The displacement of a point `x` is the increase in *model complexity*
//! (total summed depth of every other point) caused by **removing** `x` from
//! the tree.  When `x` is deleted, its sibling subtree is promoted one level
//! up, so each of the `c` points in that sibling loses exactly one unit of
//! depth — i.e. displacement `= c`.
//!
//! Plain displacement is fooled by *collusion*: duplicating an outlier many
//! times makes each copy look normal.  CoDisp fixes this by considering, along
//! the path from `x` to the root, the removal of the **whole subtree** rooted
//! at each ancestor.  Removing a subtree `S` of size `|S|` promotes its sibling
//! of size `c`, displacing `c` points; the contribution attributed to a single
//! member of `S` is `c / |S|`.  CoDisp is the maximum of `c / |S|` over the
//! path:
//!
//! ```text
//! CoDisp(x) = max_{S ∋ x on the path}  |sibling(S)| / |S|
//! ```
//!
//! The forest score is the average CoDisp over all trees.  Higher ⇒ more
//! anomalous.
//!
//! # Streaming
//!
//! [`RrcTree::insert_point`] and [`RrcTree::forget_point`] maintain the tree
//! and all bounding boxes incrementally, so the model tracks an evolving
//! stream.  Inserting a point and then forgetting it restores the previous
//! tree (identical structure and scores).

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Bounding box ───────────────────────────────────────────────────────────

/// Axis-aligned bounding box stored as per-dimension `[low, high]` pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundingBox {
    /// Per-dimension lower corner.
    pub low: Vec<f32>,
    /// Per-dimension upper corner.
    pub high: Vec<f32>,
}

impl BoundingBox {
    /// Degenerate box around a single point.
    #[must_use]
    pub fn point(x: &[f32]) -> Self {
        Self {
            low: x.to_vec(),
            high: x.to_vec(),
        }
    }

    /// Side length in dimension `i` (`high − low`).
    #[must_use]
    #[inline]
    pub fn side(&self, i: usize) -> f32 {
        self.high[i] - self.low[i]
    }

    /// Sum of all side lengths (the bounding box "span").
    #[must_use]
    #[inline]
    pub fn span(&self) -> f32 {
        self.low
            .iter()
            .zip(self.high.iter())
            .map(|(l, h)| h - l)
            .sum()
    }

    /// Smallest box enclosing both `self` and `other`.
    #[must_use]
    pub fn union(&self, other: &BoundingBox) -> BoundingBox {
        let low = self
            .low
            .iter()
            .zip(other.low.iter())
            .map(|(a, b)| a.min(*b))
            .collect();
        let high = self
            .high
            .iter()
            .zip(other.high.iter())
            .map(|(a, b)| a.max(*b))
            .collect();
        BoundingBox { low, high }
    }

    /// Extend `self` in place to also contain `x`.
    fn expand_to_point(&mut self, x: &[f32]) {
        for ((lo, hi), &xi) in self.low.iter_mut().zip(self.high.iter_mut()).zip(x.iter()) {
            if xi < *lo {
                *lo = xi;
            }
            if xi > *hi {
                *hi = xi;
            }
        }
    }
}

// ─── Node ───────────────────────────────────────────────────────────────────

/// A node in a Robust Random Cut Tree, arena-allocated in [`RrcTree::nodes`].
///
/// Internal nodes store a cut `(dim, value)` and two children plus the
/// bounding box and point count of the subtree.  Leaves store the point key,
/// its coordinates, and a multiplicity (number of identical copies collapsed
/// into one leaf — required to make duplicate insert/forget exact).
#[derive(Debug, Clone)]
pub struct RrcNode {
    /// Parent node index, or `None` for the root.
    pub parent: Option<usize>,
    /// Left child index (internal nodes only).
    pub left: Option<usize>,
    /// Right child index (internal nodes only).
    pub right: Option<usize>,
    /// Cut dimension (internal nodes only).
    pub cut_dim: usize,
    /// Cut value (internal nodes only).
    pub cut_value: f32,
    /// Bounding box of every point under this node.
    pub bbox: BoundingBox,
    /// Number of points (with multiplicity) under this node.
    pub count: usize,
    /// Leaf coordinates (`Some` only for leaves).
    pub point: Option<Vec<f32>>,
    /// Number of identical copies stored at this leaf (leaves only).
    pub multiplicity: usize,
}

impl RrcNode {
    /// Whether this node is a leaf.
    #[must_use]
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

// ─── Tree ───────────────────────────────────────────────────────────────────

/// A single Robust Random Cut Tree over a streaming point set.
#[derive(Debug, Clone)]
pub struct RrcTree {
    /// Arena of nodes; `nodes[root]` is the root when `root` is `Some`.
    pub nodes: Vec<RrcNode>,
    /// Index of the root node, or `None` when the tree is empty.
    pub root: Option<usize>,
    /// Free slots in `nodes` available for reuse after deletions.
    free: Vec<usize>,
    /// Feature dimensionality.
    dim: usize,
}

impl RrcTree {
    /// Create an empty tree over `dim`-dimensional points.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
            free: Vec::new(),
            dim,
        }
    }

    /// Number of points (with multiplicity) currently stored.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.root.map_or(0, |r| self.nodes[r].count)
    }

    /// Whether the tree currently holds no points.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Allocate a node, reusing a freed slot when available.
    fn alloc(&mut self, node: RrcNode) -> usize {
        if let Some(slot) = self.free.pop() {
            self.nodes[slot] = node;
            slot
        } else {
            self.nodes.push(node);
            self.nodes.len() - 1
        }
    }

    /// Pick the cut dimension `∝ side length` and a uniform cut value in
    /// `[low, high]` of that dimension, given a bounding box `span` budget.
    ///
    /// Returns `(dim, value)`.  Mirrors the RRCF construction exactly.
    fn choose_cut(bbox: &BoundingBox, rng: &mut LcgRng) -> (usize, f32) {
        let span = bbox.span();
        let d = bbox.low.len();
        // Degenerate span (all points identical): fall back to dim 0 at low.
        if span <= 0.0 {
            return (0, bbox.low[0]);
        }
        // Draw r ∈ [0, span); the dimension whose cumulative side length first
        // exceeds r is selected — this samples dim ∝ side length.
        let r = rng.next_f32() * span;
        let mut acc = 0.0_f32;
        let mut dim = d - 1;
        for i in 0..d {
            acc += bbox.side(i);
            if r < acc {
                dim = i;
                break;
            }
        }
        // Uniform cut value within the chosen dimension's range.
        let value = bbox.low[dim] + rng.next_f32() * bbox.side(dim);
        (dim, value)
    }

    /// Build a fresh subtree from a multiset of points given as
    /// `(coords, multiplicity)` pairs.  Returns the new subtree root index.
    fn build(&mut self, points: &[(Vec<f32>, usize)], rng: &mut LcgRng) -> usize {
        debug_assert!(!points.is_empty());

        // Compute bounding box and total multiplicity for these points.
        let mut bbox = BoundingBox::point(&points[0].0);
        let mut total = points[0].1;
        for (coords, mult) in &points[1..] {
            bbox.expand_to_point(coords);
            total += *mult;
        }

        // Leaf condition: a single distinct coordinate ⇒ collapse to one leaf.
        if points.len() == 1 {
            return self.alloc(RrcNode {
                parent: None,
                left: None,
                right: None,
                cut_dim: 0,
                cut_value: 0.0,
                bbox,
                count: total,
                point: Some(points[0].0.clone()),
                multiplicity: points[0].1,
            });
        }

        // If every coordinate is identical (zero span) but stored as separate
        // entries, collapse them into one leaf with summed multiplicity.
        if bbox.span() <= 0.0 {
            return self.alloc(RrcNode {
                parent: None,
                left: None,
                right: None,
                cut_dim: 0,
                cut_value: 0.0,
                bbox,
                count: total,
                point: Some(points[0].0.clone()),
                multiplicity: total,
            });
        }

        // Choose a cut and partition; retry until the split is non-trivial.
        let (cut_dim, cut_value, left_pts, right_pts) = loop {
            let (cd, cv) = Self::choose_cut(&bbox, rng);
            let mut lhs: Vec<(Vec<f32>, usize)> = Vec::new();
            let mut rhs: Vec<(Vec<f32>, usize)> = Vec::new();
            for (coords, mult) in points {
                if coords[cd] <= cv {
                    lhs.push((coords.clone(), *mult));
                } else {
                    rhs.push((coords.clone(), *mult));
                }
            }
            if !lhs.is_empty() && !rhs.is_empty() {
                break (cd, cv, lhs, rhs);
            }
        };

        // Reserve this internal node, then build children and back-fill links.
        let this = self.alloc(RrcNode {
            parent: None,
            left: None,
            right: None,
            cut_dim,
            cut_value,
            bbox: bbox.clone(),
            count: total,
            point: None,
            multiplicity: 0,
        });
        let l = self.build(&left_pts, rng);
        let r = self.build(&right_pts, rng);
        self.nodes[l].parent = Some(this);
        self.nodes[r].parent = Some(this);
        self.nodes[this].left = Some(l);
        self.nodes[this].right = Some(r);
        this
    }

    /// Build a complete tree from a row-major `[n × dim]` matrix.
    fn fit_matrix(&mut self, data: &[f32], n: usize, rng: &mut LcgRng) {
        let points: Vec<(Vec<f32>, usize)> = (0..n)
            .map(|i| (data[i * self.dim..(i + 1) * self.dim].to_vec(), 1))
            .collect();
        if points.is_empty() {
            return;
        }
        let root = self.build(&points, rng);
        self.root = Some(root);
    }

    /// Insert a single point, updating structure and all ancestor bounding
    /// boxes.  Duplicate coordinates increase a leaf's multiplicity in place.
    ///
    /// The cut placing the new point is drawn from the *isolation* construction
    /// of Guha et al.: at the chosen node a fresh cut is sampled in the box
    /// that already contains the existing subtree extended to include `x`; if
    /// the cut separates `x` from the subtree, a new internal node is created.
    pub fn insert_point(&mut self, x: &[f32], rng: &mut LcgRng) -> AnomalyResult<()> {
        if x.len() != self.dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        match self.root {
            None => {
                let leaf = self.alloc(RrcNode {
                    parent: None,
                    left: None,
                    right: None,
                    cut_dim: 0,
                    cut_value: 0.0,
                    bbox: BoundingBox::point(x),
                    count: 1,
                    point: Some(x.to_vec()),
                    multiplicity: 1,
                });
                self.root = Some(leaf);
                Ok(())
            }
            Some(root) => {
                let new_leaf = self.insert_below(root, x, rng);
                if let Some(nl) = new_leaf {
                    self.root = Some(nl);
                }
                Ok(())
            }
        }
    }

    /// Recursive insertion below `node`.  Returns `Some(new_subtree_root)` if
    /// the subtree root changed (a new internal node was spliced in), else
    /// `None` (the point merged into an existing leaf in place).
    fn insert_below(&mut self, node: usize, x: &[f32], rng: &mut LcgRng) -> Option<usize> {
        // Exact-duplicate fast path: merge into a coincident leaf.
        if self.nodes[node].is_leaf()
            && let Some(p) = &self.nodes[node].point
            && p.as_slice() == x
        {
            self.nodes[node].multiplicity += 1;
            self.nodes[node].count += 1;
            return None;
        }

        // Sample a cut in the box that contains the existing subtree *and* x.
        let merged = self.nodes[node].bbox.union(&BoundingBox::point(x));
        let (cut_dim, cut_value) = Self::choose_cut(&merged, rng);

        // Does this cut isolate x from the whole existing subtree?
        let x_side_left = x[cut_dim] <= cut_value;
        let sub_low = self.nodes[node].bbox.low[cut_dim];
        let sub_high = self.nodes[node].bbox.high[cut_dim];
        let subtree_all_left = sub_high <= cut_value;
        let subtree_all_right = sub_low > cut_value;

        if (x_side_left && subtree_all_right) || (!x_side_left && subtree_all_left) {
            // The cut separates x from the entire existing subtree → splice a
            // new internal node with x as a sibling leaf.
            let leaf = self.alloc(RrcNode {
                parent: None,
                left: None,
                right: None,
                cut_dim: 0,
                cut_value: 0.0,
                bbox: BoundingBox::point(x),
                count: 1,
                point: Some(x.to_vec()),
                multiplicity: 1,
            });
            let old_count = self.nodes[node].count;
            let parent = self.nodes[node].parent;
            let branch = self.alloc(RrcNode {
                parent,
                left: None,
                right: None,
                cut_dim,
                cut_value,
                bbox: merged,
                count: old_count + 1,
                point: None,
                multiplicity: 0,
            });
            if x_side_left {
                self.nodes[branch].left = Some(leaf);
                self.nodes[branch].right = Some(node);
            } else {
                self.nodes[branch].left = Some(node);
                self.nodes[branch].right = Some(leaf);
            }
            self.nodes[leaf].parent = Some(branch);
            self.nodes[node].parent = Some(branch);
            return Some(branch);
        }

        // The cut does not isolate x → descend into the matching child.
        if self.nodes[node].is_leaf() {
            // Leaf reached without isolation (coincident-range): force a split
            // on a dimension where the leaf and x differ.
            let leaf = self.alloc(RrcNode {
                parent: None,
                left: None,
                right: None,
                cut_dim: 0,
                cut_value: 0.0,
                bbox: BoundingBox::point(x),
                count: 1,
                point: Some(x.to_vec()),
                multiplicity: 1,
            });
            let old_count = self.nodes[node].count;
            let parent = self.nodes[node].parent;
            let (fd, fv, x_left) = self.forced_split(node, x);
            let branch = self.alloc(RrcNode {
                parent,
                left: None,
                right: None,
                cut_dim: fd,
                cut_value: fv,
                bbox: merged,
                count: old_count + 1,
                point: None,
                multiplicity: 0,
            });
            if x_left {
                self.nodes[branch].left = Some(leaf);
                self.nodes[branch].right = Some(node);
            } else {
                self.nodes[branch].left = Some(node);
                self.nodes[branch].right = Some(leaf);
            }
            self.nodes[leaf].parent = Some(branch);
            self.nodes[node].parent = Some(branch);
            return Some(branch);
        }

        // Internal node: recurse into the child on x's side of *this* node's cut.
        let go_left = x[self.nodes[node].cut_dim] <= self.nodes[node].cut_value;
        let child = if go_left {
            self.nodes[node].left
        } else {
            self.nodes[node].right
        };
        // An internal node always has the child on the selected side; if the
        // invariant were ever broken we leave the tree untouched rather than
        // panicking.
        if let Some(child) = child {
            let replaced = self.insert_below(child, x, rng);
            if let Some(new_child) = replaced {
                if go_left {
                    self.nodes[node].left = Some(new_child);
                } else {
                    self.nodes[node].right = Some(new_child);
                }
                self.nodes[new_child].parent = Some(node);
            }
            // Update this node's count and bounding box to include x.
            self.nodes[node].count += 1;
            self.nodes[node].bbox.expand_to_point(x);
        }
        None
    }

    /// Find a dimension where leaf `node` and `x` differ; return a separating
    /// `(dim, value, x_goes_left)`.
    ///
    /// Only ever called on a leaf (which always carries coordinates); the empty
    /// fallback simply yields a default split on dimension 0.
    fn forced_split(&self, node: usize, x: &[f32]) -> (usize, f32, bool) {
        let Some(p) = self.nodes[node].point.as_ref() else {
            return (0, x[0], true);
        };
        for d in 0..self.dim {
            if (p[d] - x[d]).abs() > 0.0 {
                let mid = 0.5 * (p[d] + x[d]);
                let x_left = x[d] <= mid;
                return (d, mid, x_left);
            }
        }
        // Identical coordinates should have been merged earlier; default split.
        (0, x[0], true)
    }

    /// Sibling of `child` under `parent`, or `None` if `parent` is malformed.
    #[inline]
    fn sibling_of(&self, parent: usize, child: usize) -> Option<usize> {
        let l = self.nodes[parent].left?;
        let r = self.nodes[parent].right?;
        if l == child { Some(r) } else { Some(l) }
    }

    /// Remove one copy of point `x`.  Returns `true` if a copy was found and
    /// removed; `false` if `x` is not in the tree.  Bounding boxes and counts
    /// of all surviving ancestors are repaired.
    pub fn forget_point(&mut self, x: &[f32]) -> AnomalyResult<bool> {
        if x.len() != self.dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let Some(root) = self.root else {
            return Ok(false);
        };
        // Locate the leaf holding x.
        let Some(leaf) = self.find_leaf(root, x) else {
            return Ok(false);
        };

        // Decrement multiplicity; if copies remain, just fix counts upward.
        if self.nodes[leaf].multiplicity > 1 {
            self.nodes[leaf].multiplicity -= 1;
            self.nodes[leaf].count -= 1;
            self.repair_ancestors(self.nodes[leaf].parent);
            return Ok(true);
        }

        // Last copy: remove the leaf, promote its sibling into the grandparent.
        match self.nodes[leaf].parent {
            None => {
                // Leaf was the root → tree becomes empty.
                self.recycle(leaf);
                self.root = None;
            }
            Some(parent) => {
                let Some(sibling) = self.sibling_of(parent, leaf) else {
                    // Malformed parent (should be impossible): nothing to do.
                    return Ok(true);
                };
                let grandparent = self.nodes[parent].parent;
                self.nodes[sibling].parent = grandparent;
                match grandparent {
                    None => self.root = Some(sibling),
                    Some(gp) => {
                        if self.nodes[gp].left == Some(parent) {
                            self.nodes[gp].left = Some(sibling);
                        } else {
                            self.nodes[gp].right = Some(sibling);
                        }
                    }
                }
                self.recycle(leaf);
                self.recycle(parent);
                self.repair_ancestors(grandparent);
            }
        }
        Ok(true)
    }

    /// Descend from `node` to the leaf whose stored coordinates equal `x`.
    fn find_leaf(&self, node: usize, x: &[f32]) -> Option<usize> {
        let mut cur = node;
        loop {
            if self.nodes[cur].is_leaf() {
                return match &self.nodes[cur].point {
                    Some(p) if p.as_slice() == x => Some(cur),
                    _ => None,
                };
            }
            let go_left = x[self.nodes[cur].cut_dim] <= self.nodes[cur].cut_value;
            cur = if go_left {
                self.nodes[cur].left?
            } else {
                self.nodes[cur].right?
            };
        }
    }

    /// Recompute `count` and `bbox` for `node` and every ancestor up to the root.
    fn repair_ancestors(&mut self, mut node: Option<usize>) {
        while let Some(n) = node {
            let (l, r) = (self.nodes[n].left, self.nodes[n].right);
            if let (Some(l), Some(r)) = (l, r) {
                self.nodes[n].count = self.nodes[l].count + self.nodes[r].count;
                self.nodes[n].bbox = self.nodes[l].bbox.union(&self.nodes[r].bbox);
            }
            node = self.nodes[n].parent;
        }
    }

    /// Mark a node slot as free for later reuse.
    fn recycle(&mut self, idx: usize) {
        self.free.push(idx);
    }

    // ── Scoring ──────────────────────────────────────────────────────────────

    /// Collusive Displacement of point `x` in this tree.
    ///
    /// Returns `0.0` if `x` is absent or the tree is empty.
    #[must_use]
    pub fn codisp(&self, x: &[f32]) -> f32 {
        let Some(root) = self.root else {
            return 0.0;
        };
        let Some(leaf) = self.find_leaf(root, x) else {
            return 0.0;
        };
        // Walk from the leaf up to the root.  At each step the current subtree
        // `S` has size `sub_size`; its sibling has size `sibling_size`.  The
        // per-point collusive displacement contributed at this level is
        // `sibling_size / sub_size`.  CoDisp is the maximum over the path.
        let mut best = 0.0_f32;
        let mut cur = leaf;
        while let Some(parent) = self.nodes[cur].parent {
            let Some(sibling) = self.sibling_of(parent, cur) else {
                break;
            };
            let sub_size = self.nodes[cur].count as f32;
            let sibling_size = self.nodes[sibling].count as f32;
            let disp = sibling_size / sub_size;
            if disp > best {
                best = disp;
            }
            cur = parent;
        }
        best
    }

    /// Depth of the leaf holding `x` (root depth = 0); `None` if absent.
    #[must_use]
    pub fn depth_of(&self, x: &[f32]) -> Option<usize> {
        let root = self.root?;
        let mut cur = root;
        let mut depth = 0_usize;
        loop {
            if self.nodes[cur].is_leaf() {
                return match &self.nodes[cur].point {
                    Some(p) if p.as_slice() == x => Some(depth),
                    _ => None,
                };
            }
            let go_left = x[self.nodes[cur].cut_dim] <= self.nodes[cur].cut_value;
            cur = if go_left {
                self.nodes[cur].left?
            } else {
                self.nodes[cur].right?
            };
            depth += 1;
        }
    }

    /// Insert `x`, read its CoDisp, then forget it again — a non-mutating query
    /// for points that are *not* part of the model.  Restores the prior tree.
    pub fn codisp_query(&mut self, x: &[f32], rng: &mut LcgRng) -> AnomalyResult<f32> {
        self.insert_point(x, rng)?;
        let score = self.codisp(x);
        self.forget_point(x)?;
        Ok(score)
    }
}

// ─── Configuration ──────────────────────────────────────────────────────────

/// Hyper-parameters for [`RobustRandomCutForest`].
#[derive(Debug, Clone)]
pub struct RrcfConfig {
    /// Number of trees in the forest (default `40`).
    pub n_trees: usize,
    /// Sub-sample size per tree (default `256`).
    pub tree_size: usize,
    /// RNG seed (default `42`).
    pub seed: u64,
}

impl Default for RrcfConfig {
    fn default() -> Self {
        Self {
            n_trees: 40,
            tree_size: 256,
            seed: 42,
        }
    }
}

// ─── Forest ─────────────────────────────────────────────────────────────────

/// Robust Random Cut Forest anomaly detector.
///
/// # Usage
///
/// ```rust,ignore
/// let mut forest = RobustRandomCutForest::new(RrcfConfig::default());
/// forest.fit(&train, n_samples, n_features)?;
/// let score = forest.score(&query)?;          // CoDisp averaged over trees
/// ```
pub struct RobustRandomCutForest {
    config: RrcfConfig,
    trees: Vec<RrcTree>,
    /// One independent RNG per tree, advanced by streaming updates.
    rngs: Vec<LcgRng>,
    n_features: usize,
    fitted: bool,
}

impl RobustRandomCutForest {
    /// Create an unfitted forest from `config`.
    #[must_use]
    pub fn new(config: RrcfConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            rngs: Vec::new(),
            n_features: 0,
            fitted: false,
        }
    }

    /// Fit the forest on `data` (row-major, `[n_samples × n_features]`).
    ///
    /// Each tree is built from an independent sub-sample of size
    /// `min(tree_size, n_samples)` drawn without replacement.
    pub fn fit(&mut self, data: &[f32], n_samples: usize, n_features: usize) -> AnomalyResult<()> {
        if n_samples == 0 {
            return Err(AnomalyError::EmptyInput);
        }
        if n_features == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        if data.len() != n_samples * n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_samples * n_features,
                got: data.len(),
            });
        }
        if self.config.n_trees == 0 {
            return Err(AnomalyError::Internal {
                msg: "n_trees must be > 0".into(),
            });
        }

        let ss = self.config.tree_size.min(n_samples).max(1);
        let mut rng = LcgRng::new(self.config.seed);

        let mut trees = Vec::with_capacity(self.config.n_trees);
        let mut rngs = Vec::with_capacity(self.config.n_trees);
        for _ in 0..self.config.n_trees {
            // Independent per-tree RNG seeded from the master stream so that
            // streaming updates stay reproducible and tree-local.
            let mut tree_rng = LcgRng::new(rng.next_u32() as u64 ^ (rng.next_u32() as u64) << 32);
            let sub = sample_without_replacement(n_samples, ss, &mut rng);
            let mut sub_data = Vec::with_capacity(ss * n_features);
            for &idx in &sub {
                sub_data.extend_from_slice(&data[idx * n_features..(idx + 1) * n_features]);
            }
            let mut tree = RrcTree::new(n_features);
            tree.fit_matrix(&sub_data, ss, &mut tree_rng);
            trees.push(tree);
            rngs.push(tree_rng);
        }

        self.trees = trees;
        self.rngs = rngs;
        self.n_features = n_features;
        self.fitted = true;
        Ok(())
    }

    /// Average CoDisp of `x` over all trees.  Higher ⇒ more anomalous.
    ///
    /// `x` is inserted into and immediately forgotten from every tree, so the
    /// model is unchanged and out-of-sample points are scored correctly.
    pub fn score(&mut self, x: &[f32]) -> AnomalyResult<f32> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let mut sum = 0.0_f32;
        for (tree, rng) in self.trees.iter_mut().zip(self.rngs.iter_mut()) {
            sum += tree.codisp_query(x, rng)?;
        }
        Ok(sum / self.trees.len() as f32)
    }

    /// Batch scoring; `x` is row-major `[n × n_features]`; returns `[n]`.
    pub fn score_batch(&mut self, x: &[f32], n: usize) -> AnomalyResult<Vec<f32>> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != n * self.n_features {
            return Err(AnomalyError::DimensionMismatch {
                expected: n * self.n_features,
                got: x.len(),
            });
        }
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let sample = x[i * self.n_features..(i + 1) * self.n_features].to_vec();
            scores.push(self.score(&sample)?);
        }
        Ok(scores)
    }

    /// Stream a new point into every tree (with reservoir-style forgetting left
    /// to the caller).  Updates structure and bounding boxes in place.
    pub fn insert_point(&mut self, x: &[f32]) -> AnomalyResult<()> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        for (tree, rng) in self.trees.iter_mut().zip(self.rngs.iter_mut()) {
            tree.insert_point(x, rng)?;
        }
        Ok(())
    }

    /// Forget a previously streamed point from every tree.
    pub fn forget_point(&mut self, x: &[f32]) -> AnomalyResult<()> {
        if !self.fitted {
            return Err(AnomalyError::NotFitted);
        }
        if x.len() != self.n_features {
            return Err(AnomalyError::FeatureCountMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        for tree in self.trees.iter_mut() {
            tree.forget_point(x)?;
        }
        Ok(())
    }

    /// Number of trees in the forest.
    #[must_use]
    #[inline]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of features the forest was fitted on (0 if unfitted).
    #[must_use]
    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Immutable access to the underlying trees (for inspection/testing).
    #[must_use]
    #[inline]
    pub fn trees(&self) -> &[RrcTree] {
        &self.trees
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Partial Fisher–Yates: sample `k` distinct indices from `0..n`.
fn sample_without_replacement(n: usize, k: usize, rng: &mut LcgRng) -> Vec<usize> {
    let k = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.next_usize(n - i);
        indices.swap(i, j);
    }
    indices.truncate(k);
    indices
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense Gaussian-ish cluster around the origin plus a far outlier row.
    fn cluster_with_outlier(n_inliers: usize, seed: u64) -> (Vec<f32>, usize) {
        let mut rng = LcgRng::new(seed);
        let mut data = Vec::with_capacity((n_inliers + 1) * 2);
        for _ in 0..n_inliers {
            data.push(rng.next_normal() * 0.2);
            data.push(rng.next_normal() * 0.2);
        }
        // Outlier far away.
        data.push(20.0);
        data.push(20.0);
        (data, n_inliers + 1)
    }

    // ── (a) A clear outlier has HIGHER CoDisp than dense inliers ──────────────
    #[test]
    fn outlier_has_higher_codisp() {
        let (data, n) = cluster_with_outlier(120, 7);
        let mut forest = RobustRandomCutForest::new(RrcfConfig {
            n_trees: 60,
            tree_size: 128,
            seed: 11,
        });
        forest.fit(&data, n, 2).expect("fit");

        let outlier = [20.0_f32, 20.0];
        let inlier = [0.0_f32, 0.0];
        let s_out = forest.score(&outlier).expect("score outlier");
        let s_in = forest.score(&inlier).expect("score inlier");
        assert!(
            s_out > s_in,
            "outlier CoDisp {s_out} should exceed inlier CoDisp {s_in}"
        );
    }

    // ── (b) Cut dimension probability ∝ bounding-box side length ──────────────
    #[test]
    fn cut_dimension_proportional_to_side_length() {
        // Box with side lengths 1.0 (dim 0) and 3.0 (dim 1) ⇒ dim 1 should be
        // chosen roughly 3× as often as dim 0.
        let bbox = BoundingBox {
            low: vec![0.0, 0.0],
            high: vec![1.0, 3.0],
        };
        let mut rng = LcgRng::new(123);
        let trials = 40_000;
        let mut counts = [0usize; 2];
        for _ in 0..trials {
            let (dim, _val) = RrcTree::choose_cut(&bbox, &mut rng);
            counts[dim] += 1;
        }
        let p0 = counts[0] as f64 / trials as f64;
        let p1 = counts[1] as f64 / trials as f64;
        // Expected: p0 = 1/4 = 0.25, p1 = 3/4 = 0.75.
        assert!((p0 - 0.25).abs() < 0.02, "p0={p0} expected ≈0.25");
        assert!((p1 - 0.75).abs() < 0.02, "p1={p1} expected ≈0.75");
    }

    // ── (c) Cut value lies within the box range ───────────────────────────────
    #[test]
    fn cut_value_within_box_range() {
        let bbox = BoundingBox {
            low: vec![-2.0, 5.0, 100.0],
            high: vec![3.0, 9.0, 110.0],
        };
        let mut rng = LcgRng::new(456);
        for _ in 0..10_000 {
            let (dim, val) = RrcTree::choose_cut(&bbox, &mut rng);
            assert!(
                val >= bbox.low[dim] && val <= bbox.high[dim],
                "cut value {val} out of range [{},{}] in dim {dim}",
                bbox.low[dim],
                bbox.high[dim]
            );
        }
    }

    // ── (d) Streaming: insert then delete restores the prior tree ─────────────
    #[test]
    fn insert_then_forget_restores_structure() {
        let mut rng = LcgRng::new(99);
        let mut tree = RrcTree::new(2);
        // Build a tree of distinct points.
        let pts = [
            [0.0_f32, 0.0],
            [1.0, 0.5],
            [0.3, 2.0],
            [2.0, 1.0],
            [-1.0, -1.0],
            [0.7, 1.3],
        ];
        for p in &pts {
            tree.insert_point(p, &mut rng).expect("insert");
        }
        let before_len = tree.len();
        // Snapshot depths of all existing points.
        let before_depths: Vec<Option<usize>> = pts.iter().map(|p| tree.depth_of(p)).collect();

        // Insert a new point, then forget it.
        let new_pt = [5.0_f32, 5.0];
        tree.insert_point(&new_pt, &mut rng).expect("insert new");
        let removed = tree.forget_point(&new_pt).expect("forget");
        assert!(removed, "point should have been removed");

        // Length and all prior depths must be restored exactly.
        assert_eq!(tree.len(), before_len, "length must be restored");
        let after_depths: Vec<Option<usize>> = pts.iter().map(|p| tree.depth_of(p)).collect();
        assert_eq!(
            before_depths, after_depths,
            "tree structure (depths) must be restored after insert+forget"
        );
        // The forgotten point must be gone.
        assert_eq!(
            tree.depth_of(&new_pt),
            None,
            "forgotten point still present"
        );
    }

    // ── (e) CoDisp ≥ 0 ────────────────────────────────────────────────────────
    #[test]
    fn codisp_non_negative() {
        let (data, n) = cluster_with_outlier(80, 3);
        let mut forest = RobustRandomCutForest::new(RrcfConfig {
            n_trees: 30,
            tree_size: 64,
            seed: 5,
        });
        forest.fit(&data, n, 2).expect("fit");
        for i in 0..n {
            let row = &data[i * 2..(i + 1) * 2];
            let s = forest.score(row).expect("score");
            assert!(s >= 0.0, "CoDisp must be ≥ 0, got {s}");
            assert!(s.is_finite(), "CoDisp must be finite, got {s}");
        }
    }

    // ── (f) Collusion: duplicated outlier — CoDisp captures collusion better ──
    #[test]
    fn codisp_handles_collusion_better_than_depth() {
        // Build a single tree: a tight inlier cluster + a *colluding* cluster of
        // many identical-ish outlier copies.  Under plain depth, the colluding
        // copies look deep (normal) because they isolate each other; CoDisp
        // still flags them because collapsing the whole colluding subtree
        // displaces the large inlier sibling.
        let mut rng = LcgRng::new(2024);
        let mut tree = RrcTree::new(2);
        // Inlier cluster.
        for _ in 0..40 {
            let p = [rng.next_normal() * 0.05, rng.next_normal() * 0.05];
            tree.insert_point(&p, &mut rng).expect("insert inlier");
        }
        // Colluding outlier cluster: several near-duplicate copies far away.
        let mut colluders = Vec::new();
        for k in 0..6 {
            let p = [10.0 + k as f32 * 1e-3, 10.0 + k as f32 * 1e-3];
            colluders.push(p);
            tree.insert_point(&p, &mut rng).expect("insert colluder");
        }

        // Depth of a colluding point vs CoDisp of the same point.
        let colluder = colluders[0];
        let depth = tree.depth_of(&colluder).expect("colluder depth");
        let codisp = tree.codisp(&colluder);

        // An inlier near the centre for comparison.
        let inlier = [0.0_f32, 0.0];
        // Inlier may not be an exact stored coordinate; query via insert/forget.
        let codisp_inlier = tree
            .codisp_query(&inlier, &mut rng)
            .expect("inlier codisp query");

        // CoDisp of the colluding cluster should exceed that of a central inlier
        // even though the colluder sits at a non-trivial depth.
        assert!(
            codisp > codisp_inlier,
            "colluder CoDisp {codisp} should exceed inlier CoDisp {codisp_inlier} (depth={depth})"
        );
        // Sanity: the colluding subtree (size 6) promotes the inlier sibling
        // (size 40) ⇒ contribution 40/6 ≈ 6.67, far above 1.
        assert!(codisp > 1.0, "colluder CoDisp {codisp} should exceed 1.0");
    }

    // ── (g) Forest averages over trees (more trees → lower score variance) ────
    #[test]
    fn more_trees_reduce_score_variance() {
        let (data, n) = cluster_with_outlier(100, 17);
        let query = [0.1_f32, -0.1];

        // Estimate score variance across several seeds for a small forest …
        let var_small = score_variance_across_seeds(&data, n, &query, 4, 64);
        // … and for a larger forest.
        let var_large = score_variance_across_seeds(&data, n, &query, 64, 64);

        assert!(
            var_large < var_small,
            "larger forest variance {var_large} should be below small forest {var_small}"
        );
    }

    /// Variance of the inlier score across 8 different RNG seeds.
    fn score_variance_across_seeds(
        data: &[f32],
        n: usize,
        query: &[f32],
        n_trees: usize,
        tree_size: usize,
    ) -> f32 {
        let mut scores = Vec::new();
        for seed in 0..8_u64 {
            let mut forest = RobustRandomCutForest::new(RrcfConfig {
                n_trees,
                tree_size,
                seed: 1000 + seed,
            });
            forest.fit(data, n, 2).expect("fit");
            scores.push(forest.score(query).expect("score"));
        }
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32
    }

    // ── Bounding box union / span sanity ──────────────────────────────────────
    #[test]
    fn bbox_union_and_span() {
        let a = BoundingBox {
            low: vec![0.0, 1.0],
            high: vec![2.0, 3.0],
        };
        let b = BoundingBox {
            low: vec![-1.0, 2.0],
            high: vec![1.0, 5.0],
        };
        let u = a.union(&b);
        assert_eq!(u.low, vec![-1.0, 1.0]);
        assert_eq!(u.high, vec![2.0, 5.0]);
        // span = (2-(-1)) + (5-1) = 3 + 4 = 7
        assert!((u.span() - 7.0).abs() < 1e-6, "span={}", u.span());
    }

    // ── Error handling ────────────────────────────────────────────────────────
    #[test]
    fn unfitted_score_errors() {
        let mut forest = RobustRandomCutForest::new(RrcfConfig::default());
        match forest.score(&[0.0_f32, 0.0]) {
            Err(AnomalyError::NotFitted) => {}
            other => panic!("expected NotFitted, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_errors() {
        let mut forest = RobustRandomCutForest::new(RrcfConfig::default());
        match forest.fit(&[], 0, 2) {
            Err(AnomalyError::EmptyInput) => {}
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }

    #[test]
    fn feature_count_mismatch_on_score() {
        let (data, n) = cluster_with_outlier(20, 1);
        let mut forest = RobustRandomCutForest::new(RrcfConfig {
            n_trees: 5,
            tree_size: 16,
            seed: 2,
        });
        forest.fit(&data, n, 2).expect("fit");
        match forest.score(&[0.0_f32, 0.0, 0.0]) {
            Err(AnomalyError::FeatureCountMismatch {
                expected: 2,
                got: 3,
            }) => {}
            other => panic!("expected FeatureCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn dimension_mismatch_on_fit() {
        let mut forest = RobustRandomCutForest::new(RrcfConfig::default());
        match forest.fit(&[1.0_f32, 2.0, 3.0], 2, 2) {
            Err(AnomalyError::DimensionMismatch {
                expected: 4,
                got: 3,
            }) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    // ── Duplicate insert/forget multiplicity ──────────────────────────────────
    #[test]
    fn duplicate_insert_forget_multiplicity() {
        let mut rng = LcgRng::new(321);
        let mut tree = RrcTree::new(1);
        tree.insert_point(&[1.0_f32], &mut rng).expect("ins1");
        tree.insert_point(&[2.0_f32], &mut rng).expect("ins2");
        // Insert the same point 3 times.
        for _ in 0..3 {
            tree.insert_point(&[2.0_f32], &mut rng).expect("dup ins");
        }
        assert_eq!(tree.len(), 5, "5 points total (with multiplicity)");
        // Forget the duplicate 3 times → back to 2 points.
        for _ in 0..3 {
            assert!(tree.forget_point(&[2.0_f32]).expect("forget dup"));
        }
        assert_eq!(tree.len(), 2, "back to 2 points");
        // One more forget removes the last copy.
        assert!(tree.forget_point(&[2.0_f32]).expect("forget last"));
        assert_eq!(tree.len(), 1, "one point left");
        assert_eq!(tree.depth_of(&[2.0_f32]), None, "all copies gone");
    }

    // ── Forget absent point returns false ─────────────────────────────────────
    #[test]
    fn forget_absent_returns_false() {
        let mut rng = LcgRng::new(7);
        let mut tree = RrcTree::new(2);
        tree.insert_point(&[0.0_f32, 0.0], &mut rng).expect("ins");
        assert!(
            !tree.forget_point(&[9.0_f32, 9.0]).expect("forget absent"),
            "forgetting an absent point should return false"
        );
        assert_eq!(tree.len(), 1, "length unchanged");
    }
}
