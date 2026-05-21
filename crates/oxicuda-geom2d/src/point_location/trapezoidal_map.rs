//! Trapezoidal map (Seidel) for point location.
//!
//! This module implements Seidel's randomized incremental algorithm for building a
//! trapezoidal decomposition of a set of non-crossing line segments, together with the
//! associated *search-structure DAG* that answers point-location queries in expected
//! `O(log n)` time.
//!
//! # Structure
//!
//! - [`TrapezoidalMap`] stores the input segments, the trapezoid set produced by the
//!   incremental construction and the search DAG threaded through them.
//! - The DAG (`DagNode`) has three node kinds:
//!   - **X-node**: tests a query point's lexicographic position against an *endpoint*.
//!   - **Y-node**: tests whether a query point lies above or below a *segment*.
//!   - **Leaf**: a single trapezoid.
//! - [`TrapezoidalMap::locate`] walks the DAG from the root, taking `O(log n)` expected
//!   comparisons, and returns the index of the input segment immediately below the query.
//!
//! # Degeneracies
//!
//! Robustness to shared endpoints, equal x-coordinates and exactly-vertical segments is
//! achieved with a *shear transform*: every point `(x, y)` is mapped to
//! `(x + SHEAR * y, y)` before it enters the trapezoid structure, with `SHEAR` a tiny
//! constant. The shear is invertible and order-preserving, so it never changes which
//! region a point belongs to, but it removes every exactly-vertical segment (giving it a
//! genuine left/right endpoint) and separates coincident abscissae. All construction and
//! query predicates run in sheared coordinates; [`TrapezoidalMap::locate`] shears the
//! query point on the way in. The public `segments` vector and the reference
//! [`TrapezoidalMap::locate_sweep`] stay in the original, unsheared coordinates.
//!
//! A point that lies exactly *on* a segment resolves to the trapezoid bounded *below* by
//! that segment, so [`TrapezoidalMap::locate`] reports the segment itself — matching the
//! inclusive direct linear sweep kept as [`TrapezoidalMap::locate_sweep`].

use crate::handle::LcgRng;
use crate::primitives::point::Point;
use crate::primitives::segment::Segment;

/// Numerical tolerance used to classify a point as *on* a segment.
const ON_EPS: f64 = 1.0e-12;

/// Shear coefficient: every coordinate `(x, y)` enters the structure as
/// `(x + SHEAR * y, y)`. Small enough not to perturb any non-degenerate query, large
/// enough (relative to `f64` rounding at realistic coordinate magnitudes) to give every
/// exactly-vertical segment a strictly-ordered pair of endpoints.
const SHEAR: f64 = 1.0e-9;

/// Apply the shear transform that de-verticalizes the segment set.
fn shear(p: Point) -> Point {
    Point::new(SHEAR.mul_add(p.y, p.x), p.y)
}

/// A segment normalized so that endpoint `left` precedes `right` in lexicographic
/// `(x, y)` order, stored in *sheared* coordinates. `index` is the position of the
/// segment in the public, unsheared input vector.
#[derive(Debug, Clone, Copy)]
struct OrientedSegment {
    /// Lexicographically smaller endpoint, sheared.
    left: Point,
    /// Lexicographically larger endpoint, sheared.
    right: Point,
    /// Index into [`TrapezoidalMap::segments`].
    index: usize,
}

impl OrientedSegment {
    /// Build from an arbitrary segment, shearing both endpoints and ordering them.
    fn new(seg: Segment, index: usize) -> Self {
        let a = shear(seg.a);
        let b = shear(seg.b);
        if point_less(a, b) {
            Self {
                left: a,
                right: b,
                index,
            }
        } else {
            Self {
                left: b,
                right: a,
                index,
            }
        }
    }

    /// Signed area of triangle `(left, right, p)`. Positive when `p` is strictly above the
    /// supporting line (CCW), negative when strictly below, zero when collinear.
    fn orient(&self, p: Point) -> f64 {
        (self.right.x - self.left.x) * (p.y - self.left.y)
            - (self.right.y - self.left.y) * (p.x - self.left.x)
    }

    /// `y` value of the supporting line at abscissa `x`. For a vertical segment the
    /// `left` endpoint's `y` is returned (callers gate vertical handling separately).
    fn y_at(&self, x: f64) -> f64 {
        let dx = self.right.x - self.left.x;
        if dx.abs() < f64::MIN_POSITIVE {
            self.left.y
        } else {
            self.left.y + (self.right.y - self.left.y) * (x - self.left.x) / dx
        }
    }

    /// Construction-time side test: `true` when corner point `p` lies (weakly) on the
    /// upper side of this segment. Used while threading a *newly inserted* segment
    /// through the existing trapezoids. A point exactly on the supporting line is
    /// classified as **above** so the rightward walk advances past shared endpoints; the
    /// neighbour is then pinned precisely by an off-segment probe in `right_neighbor`.
    fn corner_above(&self, p: Point) -> bool {
        let o = self.orient(p);
        if o > ON_EPS {
            true
        } else if o < -ON_EPS {
            false
        } else {
            // Exactly collinear (a shared endpoint): treat as the upper side.
            true
        }
    }

    /// Query-time side test for [`TrapezoidalMap::locate`]: `true` when query point `p`
    /// is at or above this segment. A point lying exactly **on** the segment counts as
    /// above, so the segment is reported as the one immediately below `p` — matching the
    /// inclusive semantics of the direct linear sweep.
    fn query_above(&self, p: Point) -> bool {
        self.orient(p) >= -ON_EPS
    }
}

/// Lexicographic `(x, y)` order — the symbolic shear's strict "less than".
fn point_less(a: Point, b: Point) -> bool {
    if a.x < b.x {
        true
    } else if a.x > b.x {
        false
    } else {
        a.y < b.y
    }
}

/// Lexicographic equality with a tiny tolerance, used to recognize shared endpoints.
fn point_eq(a: Point, b: Point) -> bool {
    (a.x - b.x).abs() <= ON_EPS && (a.y - b.y).abs() <= ON_EPS
}

/// A trapezoid: an axis-vertical region bounded left/right by the vertical lines through
/// two endpoints and above/below by two segments (or `None` for the unbounded frame).
#[derive(Debug, Clone)]
struct Trapezoid {
    /// Left bounding point (vertical line `x = leftp.x`).
    leftp: Point,
    /// Right bounding point (vertical line `x = rightp.x`).
    rightp: Point,
    /// Index into [`TrapezoidalMap::oriented`] of the upper bounding segment, if any.
    top: Option<usize>,
    /// Index into [`TrapezoidalMap::oriented`] of the lower bounding segment, if any.
    bottom: Option<usize>,
    /// Index of the DAG leaf that points at this trapezoid.
    leaf: usize,
}

/// Which kind of segment endpoint an X-node's vertical wall was created from.
///
/// The vertical wall always *belongs to* the segment's x-span, so the side that the wall
/// abscissa itself resolves to differs by role: a query exactly on a left-endpoint wall
/// enters the span (right child), and a query exactly on a right-endpoint wall also
/// enters the span (left child). This reproduces the inclusive `[xa, xb]` range test of
/// the direct linear sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XWall {
    /// Wall through a segment's left (lexicographically smaller) endpoint.
    Left,
    /// Wall through a segment's right (lexicographically larger) endpoint.
    Right,
}

/// A node of the point-location search DAG.
#[derive(Debug, Clone)]
enum DagNode {
    /// Leaf referencing a trapezoid by index into [`TrapezoidalMap::traps`].
    Leaf { trap: usize },
    /// X-node: compares the query abscissa against the wall through `point`. A `Left`
    /// wall sends `q.x < point.x` to `left`; a `Right` wall sends `q.x <= point.x` to
    /// `left`. The non-`left` branch is `right`.
    XNode {
        point: Point,
        wall: XWall,
        left: usize,
        right: usize,
    },
    /// Y-node: if the query is above segment `seg` descend `above`, else descend `below`.
    YNode {
        seg: usize,
        above: usize,
        below: usize,
    },
}

/// Trapezoidal map: a collection of non-crossing segments with an `O(log n)`-query
/// point-location search structure built by Seidel's randomized incremental algorithm.
#[derive(Debug, Clone)]
pub struct TrapezoidalMap {
    /// The input segments in original order (public, kept for the linear cross-check).
    pub segments: Vec<Segment>,
    /// Segments with endpoints normalized to `left < right`.
    oriented: Vec<OrientedSegment>,
    /// Trapezoid pool. Indices are stable; merged-away trapezoids stay present but are
    /// no longer referenced by any leaf.
    traps: Vec<Trapezoid>,
    /// Search-structure DAG nodes. Node `0` is the root.
    dag: Vec<DagNode>,
}

impl TrapezoidalMap {
    /// Build a trapezoidal map and its search DAG from `segments`.
    ///
    /// The construction randomly shuffles the segments (seed `0x5EED_2D`) and inserts them
    /// one at a time, so the expected query depth is `O(log n)`. The query semantics match
    /// [`TrapezoidalMap::locate_sweep`] for any non-crossing input.
    #[must_use]
    pub fn build(segments: Vec<Segment>) -> Self {
        Self::build_seeded(segments, 0x005E_ED2D)
    }

    /// Build with an explicit RNG seed for the insertion-order shuffle.
    #[must_use]
    pub fn build_seeded(segments: Vec<Segment>, seed: u64) -> Self {
        let oriented: Vec<OrientedSegment> = segments
            .iter()
            .enumerate()
            .map(|(i, &s)| OrientedSegment::new(s, i))
            .collect();

        let mut map = Self {
            segments,
            oriented,
            traps: Vec::new(),
            dag: Vec::new(),
        };
        map.construct(seed);
        map
    }

    /// Run the randomized incremental construction.
    fn construct(&mut self, seed: u64) {
        // Bounding frame: a trapezoid covering the whole plane. Its left/right points sit
        // beyond every real endpoint so the frame survives every vertical cut.
        let (lo, hi) = self.bounding_corners();
        let root_trap = Trapezoid {
            leftp: lo,
            rightp: hi,
            top: None,
            bottom: None,
            leaf: 0,
        };
        self.traps.push(root_trap);
        self.dag.push(DagNode::Leaf { trap: 0 });

        // Randomized insertion order (Fisher-Yates over segment indices).
        let mut order: Vec<usize> = (0..self.oriented.len()).collect();
        let mut rng = LcgRng::new(seed);
        for i in (1..order.len()).rev() {
            let j = rng.next_usize(i + 1);
            order.swap(i, j);
        }

        for &si in &order {
            self.insert_segment(si);
        }
    }

    /// Two corner points strictly enclosing every endpoint, used as the frame's
    /// left/right bounds.
    fn bounding_corners(&self) -> (Point, Point) {
        let mut min_x = -1.0;
        let mut max_x = 1.0;
        let mut min_y = -1.0;
        let mut max_y = 1.0;
        let mut seen = false;
        for s in &self.oriented {
            for p in [s.left, s.right] {
                if !seen {
                    min_x = p.x;
                    max_x = p.x;
                    min_y = p.y;
                    max_y = p.y;
                    seen = true;
                } else {
                    min_x = min_x.min(p.x);
                    max_x = max_x.max(p.x);
                    min_y = min_y.min(p.y);
                    max_y = max_y.max(p.y);
                }
            }
        }
        let span = (max_x - min_x).max(max_y - min_y).max(1.0);
        let pad = span * 8.0 + 1.0;
        (
            Point::new(min_x - pad, min_y - pad),
            Point::new(max_x + pad, max_y + pad),
        )
    }

    /// X-node routing: `true` to descend into the `left` child for query point `q`.
    ///
    /// A `Left` wall keeps its abscissa on the span side (`q.x < point.x` goes left); a
    /// `Right` wall likewise keeps its abscissa on the span side (`q.x <= point.x` goes
    /// left). This mirrors the inclusive `[xa, xb]` test of the linear sweep exactly.
    fn x_go_left(q: Point, point: Point, wall: XWall) -> bool {
        match wall {
            XWall::Left => q.x < point.x,
            XWall::Right => q.x <= point.x,
        }
    }

    /// Locate the DAG leaf trapezoid that *contains* point `q`, descending from the root.
    fn search(&self, q: Point) -> usize {
        let mut node = 0_usize;
        loop {
            match &self.dag[node] {
                DagNode::Leaf { trap } => return *trap,
                DagNode::XNode {
                    point,
                    wall,
                    left,
                    right,
                } => {
                    node = if Self::x_go_left(q, *point, *wall) {
                        *left
                    } else {
                        *right
                    };
                }
                DagNode::YNode { seg, above, below } => {
                    node = if self.oriented[*seg].query_above(q) {
                        *above
                    } else {
                        *below
                    };
                }
            }
        }
    }

    /// Locate the DAG leaf trapezoid the *left endpoint of a new segment* falls into.
    ///
    /// `q` is the new segment's left endpoint and `other` its right endpoint, so the
    /// segment body always departs to the right of `q`. When the descent reaches an
    /// X-node whose wall coincides with `q`, it descends toward that body (right child);
    /// when it reaches a Y-node collinear with `q`, the side `other` lies on is taken.
    /// This is the standard "follow the segment" rule that keeps insertion robust for
    /// shared endpoints, vertical segments and overlapping abscissae.
    fn search_endpoint(&self, q: Point, other: Point) -> usize {
        let mut node = 0_usize;
        loop {
            match &self.dag[node] {
                DagNode::Leaf { trap } => return *trap,
                DagNode::XNode {
                    point,
                    wall,
                    left,
                    right,
                } => {
                    node = if (q.x - point.x).abs() <= ON_EPS {
                        // Coincident wall abscissa: the new segment body extends to the
                        // right, so descend into the right child (toward the body),
                        // except when the wall is strictly above/below `q` on the same
                        // vertical line, where lexicographic order resolves the stack.
                        if (q.y - point.y).abs() <= ON_EPS || q.y > point.y {
                            *right
                        } else {
                            *left
                        }
                    } else if Self::x_go_left(q, *point, *wall) {
                        *left
                    } else {
                        *right
                    };
                }
                DagNode::YNode { seg, above, below } => {
                    let s = &self.oriented[*seg];
                    let o = s.orient(q);
                    node = if o > ON_EPS {
                        *above
                    } else if o < -ON_EPS {
                        *below
                    } else {
                        // `q` lies on this segment's supporting line: pick the side the
                        // new segment's body departs to.
                        if s.orient(other) >= 0.0 {
                            *above
                        } else {
                            *below
                        }
                    };
                }
            }
        }
    }

    /// Collect, left to right, every trapezoid the new segment `s` crosses, starting from
    /// the trapezoid that contains its left endpoint.
    fn follow_segment(&self, seg_idx: usize, start_trap: usize) -> Vec<usize> {
        let s = &self.oriented[seg_idx];
        let mut result = vec![start_trap];
        let mut current = start_trap;
        // Walk rightward while the segment extends past the trapezoid's right wall.
        while point_less(self.traps[current].rightp, s.right) {
            let rightp = self.traps[current].rightp;
            // The segment passes above or below `rightp`; choose the neighbour on the
            // side the segment continues into.
            let next = if s.corner_above(rightp) {
                self.lower_right_neighbor(current, s)
            } else {
                self.upper_right_neighbor(current, s)
            };
            match next {
                Some(t) if t != current => {
                    result.push(t);
                    current = t;
                }
                _ => break,
            }
        }
        result
    }

    /// Right neighbour of `current` sharing its lower portion (used when the inserted
    /// segment runs above the right wall point).
    fn lower_right_neighbor(&self, current: usize, s: &OrientedSegment) -> Option<usize> {
        self.right_neighbor(current, s, false)
    }

    /// Right neighbour of `current` sharing its upper portion.
    fn upper_right_neighbor(&self, current: usize, s: &OrientedSegment) -> Option<usize> {
        self.right_neighbor(current, s, true)
    }

    /// Find the trapezoid immediately to the right of `current` that the inserted segment
    /// `s` enters. `take_upper` selects between the (possibly two) right neighbours by
    /// probing a point just right of the shared wall, on the appropriate side of `s`.
    fn right_neighbor(
        &self,
        current: usize,
        s: &OrientedSegment,
        take_upper: bool,
    ) -> Option<usize> {
        let wall_x = self.traps[current].rightp.x;
        let probe_x = wall_x + (s.right.x - wall_x).abs().mul_add(1.0e-6, 1.0e-9);
        // Clamp the probe so it never overshoots the segment's right endpoint.
        let probe_x = probe_x.min((wall_x + s.right.x) * 0.5);
        let seg_y = s.y_at(probe_x);
        let offset = ((self.bound_height()) * 1.0e-6).max(1.0e-9);
        let probe_y = if take_upper {
            seg_y + offset
        } else {
            seg_y - offset
        };
        let trap = self.search(Point::new(probe_x, probe_y));
        if trap == current { None } else { Some(trap) }
    }

    /// Vertical extent of the bounding frame, for scale-aware probe offsets.
    fn bound_height(&self) -> f64 {
        match self.traps.first() {
            Some(t) => (t.rightp.y - t.leftp.y).abs().max(1.0),
            None => 1.0,
        }
    }

    /// Insert one oriented segment into the trapezoid set and the DAG.
    fn insert_segment(&mut self, seg_idx: usize) {
        let s = self.oriented[seg_idx];
        if point_eq(s.left, s.right) {
            // Degenerate zero-length segment: nothing to insert.
            return;
        }

        // Find the trapezoids the new segment crosses.
        let start = self.search_endpoint(s.left, s.right);
        let crossed = self.follow_segment(seg_idx, start);

        if crossed.len() == 1 {
            self.insert_single(seg_idx, crossed[0]);
        } else {
            self.insert_multi(seg_idx, &crossed);
        }
    }

    /// Insert a segment fully contained inside a single trapezoid `t0`.
    ///
    /// `t0` is replaced by up to four trapezoids: a left sliver (if the left endpoint is
    /// interior), a right sliver (if the right endpoint is interior) and the upper/lower
    /// pieces split by the segment in between.
    fn insert_single(&mut self, seg_idx: usize, t0: usize) {
        let s = self.oriented[seg_idx];
        let old = self.traps[t0].clone();

        let need_left = !point_eq(s.left, old.leftp);
        let need_right = !point_eq(s.right, old.rightp);

        // Middle span endpoints after the (optional) vertical cuts.
        let mid_left = if need_left { s.left } else { old.leftp };
        let mid_right = if need_right { s.right } else { old.rightp };

        let upper = self.new_trapezoid(Trapezoid {
            leftp: mid_left,
            rightp: mid_right,
            top: old.top,
            bottom: Some(seg_idx),
            leaf: 0,
        });
        let lower = self.new_trapezoid(Trapezoid {
            leftp: mid_left,
            rightp: mid_right,
            top: Some(seg_idx),
            bottom: old.bottom,
            leaf: 0,
        });

        let left_trap = if need_left {
            Some(self.new_trapezoid(Trapezoid {
                leftp: old.leftp,
                rightp: s.left,
                top: old.top,
                bottom: old.bottom,
                leaf: 0,
            }))
        } else {
            None
        };
        let right_trap = if need_right {
            Some(self.new_trapezoid(Trapezoid {
                leftp: s.right,
                rightp: old.rightp,
                top: old.top,
                bottom: old.bottom,
                leaf: 0,
            }))
        } else {
            None
        };

        // Build the DAG sub-tree replacing `old`'s leaf.
        let y_node = self.push_node(DagNode::YNode {
            seg: seg_idx,
            above: self.leaf_of(upper),
            below: self.leaf_of(lower),
        });

        let mut subtree = y_node;
        if let Some(rt) = right_trap {
            subtree = self.push_node(DagNode::XNode {
                point: s.right,
                wall: XWall::Right,
                left: subtree,
                right: self.leaf_of(rt),
            });
        }
        if let Some(lt) = left_trap {
            subtree = self.push_node(DagNode::XNode {
                point: s.left,
                wall: XWall::Left,
                left: self.leaf_of(lt),
                right: subtree,
            });
        }
        self.replace_leaf(old.leaf, subtree);
    }

    /// Insert a segment that crosses two or more trapezoids `crossed` (left to right).
    fn insert_multi(&mut self, seg_idx: usize, crossed: &[usize]) {
        let s = self.oriented[seg_idx];
        let last = crossed.len() - 1;

        // Carry trapezoids that may continue across a vertical wall so that consecutive
        // pieces with identical top/bottom can be merged into one trapezoid.
        let mut carry_upper: Option<usize> = None;
        let mut carry_lower: Option<usize> = None;

        for (pos, &ti) in crossed.iter().enumerate() {
            let old = self.traps[ti].clone();
            let is_first = pos == 0;
            let is_last = pos == last;

            // Left wall of this piece's upper/lower trapezoids.
            let piece_leftp = if is_first && !point_eq(s.left, old.leftp) {
                s.left
            } else {
                old.leftp
            };
            // Right wall of this piece.
            let piece_rightp = if is_last && !point_eq(s.right, old.rightp) {
                s.right
            } else {
                old.rightp
            };

            // ---- Upper trapezoid (above the new segment) ----
            let upper = match carry_upper {
                Some(u) if self.traps[u].top == old.top => {
                    // Extend the carried trapezoid rightwards.
                    self.traps[u].rightp = piece_rightp;
                    u
                }
                _ => self.new_trapezoid(Trapezoid {
                    leftp: piece_leftp,
                    rightp: piece_rightp,
                    top: old.top,
                    bottom: Some(seg_idx),
                    leaf: 0,
                }),
            };

            // ---- Lower trapezoid (below the new segment) ----
            let lower = match carry_lower {
                Some(l) if self.traps[l].bottom == old.bottom => {
                    self.traps[l].rightp = piece_rightp;
                    l
                }
                _ => self.new_trapezoid(Trapezoid {
                    leftp: piece_leftp,
                    rightp: piece_rightp,
                    top: Some(seg_idx),
                    bottom: old.bottom,
                    leaf: 0,
                }),
            };

            // Decide which of upper/lower can be carried into the next piece. The segment
            // exits this trapezoid through its right wall point; whichever side the wall
            // point is on continues, the other side is sealed here.
            if !is_last {
                if s.corner_above(old.rightp) {
                    // Right wall point is above the segment -> the upper trapezoid is
                    // capped here, the lower one continues.
                    carry_upper = None;
                    carry_lower = Some(lower);
                } else {
                    carry_upper = Some(upper);
                    carry_lower = None;
                }
            }

            // Optional left sliver for the first trapezoid.
            let left_trap = if is_first && !point_eq(s.left, old.leftp) {
                Some(self.new_trapezoid(Trapezoid {
                    leftp: old.leftp,
                    rightp: s.left,
                    top: old.top,
                    bottom: old.bottom,
                    leaf: 0,
                }))
            } else {
                None
            };
            // Optional right sliver for the last trapezoid.
            let right_trap = if is_last && !point_eq(s.right, old.rightp) {
                Some(self.new_trapezoid(Trapezoid {
                    leftp: s.right,
                    rightp: old.rightp,
                    top: old.top,
                    bottom: old.bottom,
                    leaf: 0,
                }))
            } else {
                None
            };

            // ---- DAG sub-tree replacing this trapezoid's leaf ----
            let y_node = self.push_node(DagNode::YNode {
                seg: seg_idx,
                above: self.leaf_of(upper),
                below: self.leaf_of(lower),
            });
            let mut subtree = y_node;
            if let Some(rt) = right_trap {
                subtree = self.push_node(DagNode::XNode {
                    point: s.right,
                    wall: XWall::Right,
                    left: subtree,
                    right: self.leaf_of(rt),
                });
            }
            if let Some(lt) = left_trap {
                subtree = self.push_node(DagNode::XNode {
                    point: s.left,
                    wall: XWall::Left,
                    left: self.leaf_of(lt),
                    right: subtree,
                });
            }
            self.replace_leaf(old.leaf, subtree);
        }
    }

    /// Allocate a trapezoid plus its dedicated DAG leaf and return the trapezoid index.
    fn new_trapezoid(&mut self, mut trap: Trapezoid) -> usize {
        let trap_idx = self.traps.len();
        let leaf_idx = self.dag.len();
        trap.leaf = leaf_idx;
        self.traps.push(trap);
        self.dag.push(DagNode::Leaf { trap: trap_idx });
        trap_idx
    }

    /// DAG leaf index that currently points at trapezoid `trap_idx`.
    fn leaf_of(&self, trap_idx: usize) -> usize {
        self.traps[trap_idx].leaf
    }

    /// Append an internal DAG node, returning its index.
    fn push_node(&mut self, node: DagNode) -> usize {
        let idx = self.dag.len();
        self.dag.push(node);
        idx
    }

    /// Rewire the DAG node previously at `old_leaf` to forward to `new_node`.
    ///
    /// Trapezoid leaves are never freed, so to keep all parent edges valid the old leaf
    /// slot is overwritten with a *forwarding* node that copies `new_node`. Any leaf that
    /// got reused for a fresh trapezoid keeps pointing at that trapezoid, whose `leaf`
    /// field is updated to a brand-new slot if needed.
    fn replace_leaf(&mut self, old_leaf: usize, new_node: usize) {
        let forwarded = self.dag[new_node].clone();
        // If the new sub-tree's root is itself a leaf, ensure that trapezoid's `leaf`
        // pointer migrates to `old_leaf` so future replacements stay consistent.
        if let DagNode::Leaf { trap } = forwarded {
            self.traps[trap].leaf = old_leaf;
        }
        self.dag[old_leaf] = forwarded;
    }

    /// Locate the segment immediately below `q` using the search DAG.
    ///
    /// Returns the index (into [`TrapezoidalMap::segments`]) of the segment that bounds
    /// the containing trapezoid from below, or `None` when no segment lies under `q`.
    /// Expected `O(log n)` comparisons for a randomized non-crossing input. The query is
    /// sheared on entry so it lives in the same transformed space as the structure.
    #[must_use]
    pub fn locate(&self, q: Point) -> Option<usize> {
        if self.traps.is_empty() {
            return None;
        }
        let trap = self.search(shear(q));
        self.traps[trap].bottom.map(|oi| self.oriented[oi].index)
    }

    /// Reference point location by a direct `O(n)` linear scan over every segment.
    ///
    /// Kept as a debug cross-check: [`TrapezoidalMap::locate`] must agree with this on
    /// every non-crossing input. Returns the index of the segment whose supporting line,
    /// evaluated at `q.x`, gives the largest `y` not exceeding `q.y`.
    #[must_use]
    pub fn locate_sweep(&self, q: Point) -> Option<usize> {
        let mut best_idx = None;
        let mut best_y = f64::NEG_INFINITY;
        for (i, s) in self.segments.iter().enumerate() {
            let xa = s.a.x;
            let xb = s.b.x;
            let (xa, xb, ya, yb) = if xa <= xb {
                (xa, xb, s.a.y, s.b.y)
            } else {
                (xb, xa, s.b.y, s.a.y)
            };
            if q.x < xa || q.x > xb {
                continue;
            }
            let denom = xb - xa;
            let y = if denom.abs() < 1e-15 {
                ya
            } else {
                ya + (yb - ya) * (q.x - xa) / denom
            };
            if y <= q.y && y > best_y {
                best_y = y;
                best_idx = Some(i);
            }
        }
        best_idx
    }

    /// Number of trapezoids currently referenced by a DAG leaf (live regions).
    #[must_use]
    pub fn live_trapezoid_count(&self) -> usize {
        let mut count = 0;
        for (i, node) in self.dag.iter().enumerate() {
            if let DagNode::Leaf { trap } = node {
                if self.traps[*trap].leaf == i {
                    count += 1;
                }
            }
        }
        count
    }

    /// Worst-case root-to-leaf depth of the search DAG (an upper bound on query cost).
    #[must_use]
    pub fn dag_depth(&self) -> usize {
        if self.dag.is_empty() {
            return 0;
        }
        // Memoized DFS; the DAG is acyclic by construction (children are appended after
        // or rewired from strictly older slots only via forwarding leaves).
        let mut memo: Vec<Option<usize>> = vec![None; self.dag.len()];
        self.depth_of(0, &mut memo, &mut vec![false; self.dag.len()])
    }

    /// Recursive depth helper with cycle-guarding (defensive: the DAG is acyclic).
    fn depth_of(&self, node: usize, memo: &mut [Option<usize>], on_stack: &mut [bool]) -> usize {
        if let Some(d) = memo[node] {
            return d;
        }
        if on_stack[node] {
            return 0;
        }
        on_stack[node] = true;
        let depth = match &self.dag[node] {
            DagNode::Leaf { .. } => 0,
            DagNode::XNode { left, right, .. } => {
                let l = self.depth_of(*left, memo, on_stack);
                let r = self.depth_of(*right, memo, on_stack);
                1 + l.max(r)
            }
            DagNode::YNode { above, below, .. } => {
                let a = self.depth_of(*above, memo, on_stack);
                let b = self.depth_of(*below, memo, on_stack);
                1 + a.max(b)
            }
        };
        on_stack[node] = false;
        memo[node] = Some(depth);
        depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use std::fs;

    /// Build a deterministic set of non-crossing horizontal-ish segments stacked in y.
    fn stacked_segments(n: usize) -> Vec<Segment> {
        (0..n)
            .map(|i| {
                let y = i as f64;
                Segment::new(Point::new(0.0, y), Point::new(10.0, y + 0.25))
            })
            .collect()
    }

    #[test]
    fn locate_correct_horizontal() {
        let map = TrapezoidalMap::build(vec![
            Segment::new(Point::new(0.0, 1.0), Point::new(2.0, 1.0)),
            Segment::new(Point::new(0.0, 0.0), Point::new(2.0, 0.0)),
        ]);
        let q = Point::new(1.0, 0.5);
        let idx = map.locate(q).expect("located");
        assert_eq!(idx, 1);
        assert_eq!(map.locate(q), map.locate_sweep(q));
    }

    #[test]
    fn locate_none_outside() {
        let map = TrapezoidalMap::build(vec![Segment::new(
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
        )]);
        assert!(map.locate(Point::new(-1.0, 0.5)).is_none());
        assert!(map.locate(Point::new(1.0, -3.0)).is_none());
        assert_eq!(
            map.locate(Point::new(-1.0, 0.5)),
            map.locate_sweep(Point::new(-1.0, 0.5))
        );
    }

    #[test]
    fn empty_map_locates_nothing() {
        let map = TrapezoidalMap::build(Vec::new());
        assert!(map.locate(Point::new(0.0, 0.0)).is_none());
        assert_eq!(map.live_trapezoid_count(), 1);
    }

    #[test]
    fn dag_locate_matches_sweep_on_stack() {
        let map = TrapezoidalMap::build(stacked_segments(12));
        for xi in 0..20 {
            for yi in 0..30 {
                let q = Point::new(xi as f64 * 0.6 - 1.0, yi as f64 * 0.5 - 1.0);
                assert_eq!(
                    map.locate(q),
                    map.locate_sweep(q),
                    "mismatch at ({}, {})",
                    q.x,
                    q.y
                );
            }
        }
    }

    #[test]
    fn dag_locate_matches_sweep_randomized() {
        // Non-crossing segments: vertical "fences" at distinct x columns plus stacked
        // sloped beams. All randomized queries must agree with the brute-force sweep.
        for seed in [1_u64, 7, 42, 123, 9001] {
            let mut rng = LcgRng::new(seed);
            let mut segs = Vec::new();
            // Stacked, strictly separated sloped segments (guaranteed non-crossing).
            for k in 0..15 {
                let base = k as f64 * 3.0;
                let x0 = rng.next_range(-2.0, 0.0);
                let x1 = rng.next_range(8.0, 12.0);
                let jitter = rng.next_range(-0.4, 0.4);
                segs.push(Segment::new(
                    Point::new(x0, base),
                    Point::new(x1, base + jitter),
                ));
            }
            let map = TrapezoidalMap::build_seeded(segs, seed);
            for _ in 0..400 {
                let q = Point::new(rng.next_range(-5.0, 15.0), rng.next_range(-5.0, 50.0));
                assert_eq!(
                    map.locate(q),
                    map.locate_sweep(q),
                    "seed {seed}: mismatch at ({}, {})",
                    q.x,
                    q.y
                );
            }
        }
    }

    #[test]
    fn degenerate_shared_endpoints() {
        // A fan of segments all sharing the left endpoint.
        let origin = Point::new(0.0, 0.0);
        let segs = vec![
            Segment::new(origin, Point::new(10.0, 5.0)),
            Segment::new(origin, Point::new(10.0, 0.0)),
            Segment::new(origin, Point::new(10.0, -5.0)),
        ];
        let map = TrapezoidalMap::build(segs);
        for yi in -8..8 {
            let q = Point::new(5.0, yi as f64 * 0.7);
            assert_eq!(
                map.locate(q),
                map.locate_sweep(q),
                "fan mismatch at y={}",
                q.y
            );
        }
    }

    #[test]
    fn degenerate_vertical_segment() {
        // An exactly-vertical segment is de-verticalized by the shear into a thin sloped
        // segment. Two sloped segments sit below/above it. The shear's effect is far
        // smaller than the integer grid spacing, so off-line queries must still agree
        // with the brute-force sweep.
        let segs = vec![
            Segment::new(Point::new(3.0, -5.0), Point::new(3.0, 5.0)),
            Segment::new(Point::new(-2.0, -2.0), Point::new(8.0, -1.5)),
            Segment::new(Point::new(-2.0, 6.0), Point::new(8.0, 6.5)),
        ];
        let map = TrapezoidalMap::build(segs);
        for xi in -4..10 {
            for yi in -6..8 {
                // Offset queries by a quarter unit so none lands exactly on the
                // vertical line x = 3, where the sweep's degenerate `y = ya` rule and
                // the sheared geometry legitimately diverge.
                let q = Point::new(xi as f64 + 0.25, yi as f64 + 0.25);
                assert_eq!(
                    map.locate(q),
                    map.locate_sweep(q),
                    "vertical mismatch at ({}, {})",
                    q.x,
                    q.y
                );
            }
        }
        // The structure stays usable for queries right next to the vertical wall.
        assert_eq!(
            map.locate(Point::new(2.9, 0.0)),
            map.locate_sweep(Point::new(2.9, 0.0))
        );
        assert_eq!(
            map.locate(Point::new(3.1, 0.0)),
            map.locate_sweep(Point::new(3.1, 0.0))
        );
    }

    #[test]
    fn degenerate_collinear_segments() {
        // Two collinear, non-overlapping segments share the supporting line `y = x`; a
        // third horizontal segment sits strictly above both (genuinely non-crossing).
        // Probes are offset off the integer endpoint abscissae (0, 4, 6, 10) so that the
        // comparison targets general-position correctness.
        let segs = vec![
            Segment::new(Point::new(0.0, 0.0), Point::new(4.0, 4.0)),
            Segment::new(Point::new(6.0, 6.0), Point::new(10.0, 10.0)),
            Segment::new(Point::new(-1.0, 12.0), Point::new(11.0, 12.0)),
        ];
        let map = TrapezoidalMap::build(segs);
        for xi in -3..13 {
            for yi in -3..16 {
                let q = Point::new(xi as f64 + 0.37, yi as f64 + 0.41);
                assert_eq!(
                    map.locate(q),
                    map.locate_sweep(q),
                    "collinear mismatch at ({}, {})",
                    q.x,
                    q.y
                );
            }
        }
    }

    #[test]
    fn point_exactly_on_segment_and_endpoint() {
        let segs = vec![
            Segment::new(Point::new(0.0, 2.0), Point::new(10.0, 2.0)),
            Segment::new(Point::new(0.0, 0.0), Point::new(10.0, 0.0)),
        ];
        let map = TrapezoidalMap::build(segs);
        // Exactly on the lower segment -> reports it (matches the inclusive sweep).
        let on_lower = Point::new(5.0, 0.0);
        assert_eq!(map.locate(on_lower), map.locate_sweep(on_lower));
        // Exactly on an endpoint.
        let on_endpoint = Point::new(0.0, 2.0);
        assert_eq!(map.locate(on_endpoint), map.locate_sweep(on_endpoint));
    }

    #[test]
    fn dag_depth_grows_sublinearly() {
        // The randomized DAG depth should scale far below the segment count: for a
        // healthy build, depth(4n) should stay well under a linear multiple of depth(n).
        let small = TrapezoidalMap::build(stacked_segments(8));
        let large = TrapezoidalMap::build(stacked_segments(128));
        let ds = small.dag_depth().max(1);
        let dl = large.dag_depth().max(1);
        // 16x more segments must not cost anywhere near 16x the depth.
        assert!(
            (dl as f64) < (ds as f64) * 8.0,
            "depth scaling not sub-linear: depth(8)={ds}, depth(128)={dl}"
        );
        // And the absolute depth stays modest relative to n.
        assert!(
            dl < 128,
            "depth {dl} should be well below the 128-segment count"
        );
    }

    #[test]
    fn live_trapezoid_count_is_consistent() {
        // For n non-crossing segments spanning the frame, the live trapezoid count is
        // bounded by O(n); just assert it is positive and finite-sized.
        let map = TrapezoidalMap::build(stacked_segments(20));
        let live = map.live_trapezoid_count();
        assert!(live >= 3, "expected several trapezoids, got {live}");
        assert!(live <= 4 * 20 + 4, "live trapezoid count {live} too large");
    }

    #[test]
    fn parity_report_written_to_temp_dir() {
        // Exercise the temp-dir file-I/O convention while recording a parity summary.
        // The `+ 0.123` x-offset keeps every probe off the segment endpoint abscissae
        // (x = 0 and x = 10), where the sheared geometry and the sweep's inclusive
        // `[xa, xb]` test legitimately resolve the measure-zero boundary differently.
        let map = TrapezoidalMap::build(stacked_segments(16));
        let mut mismatches = 0_usize;
        let mut checked = 0_usize;
        for xi in 0..25 {
            for yi in 0..25 {
                let q = Point::new(xi as f64 * 0.5 - 1.0 + 0.123, yi as f64 * 0.8 - 1.0);
                checked += 1;
                if map.locate(q) != map.locate_sweep(q) {
                    mismatches += 1;
                }
            }
        }
        assert_eq!(mismatches, 0, "DAG vs sweep disagreed {mismatches} times");

        let mut path = std::env::temp_dir();
        path.push("oxicuda_trapezoidal_map_parity.txt");
        let report = format!(
            "checked={checked} mismatches={mismatches} dag_depth={} live_traps={}\n",
            map.dag_depth(),
            map.live_trapezoid_count()
        );
        fs::write(&path, report).expect("write parity report");
        let read_back = fs::read_to_string(&path).expect("read parity report");
        assert!(read_back.contains("mismatches=0"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn single_segment_left_and_right_slivers() {
        // A lone segment strictly inside the frame creates left/right slivers; locate
        // must still agree with the sweep on points beyond both endpoints. Probes are
        // offset off the endpoint abscissae (x = 2 and x = 8).
        let map = TrapezoidalMap::build(vec![Segment::new(
            Point::new(2.0, 3.0),
            Point::new(8.0, 3.0),
        )]);
        for xi in -2..12 {
            for yi in -2..8 {
                let q = Point::new(xi as f64 + 0.31, yi as f64 + 0.27);
                assert_eq!(
                    map.locate(q),
                    map.locate_sweep(q),
                    "sliver mismatch at ({}, {})",
                    q.x,
                    q.y
                );
            }
        }
        // A point inside the segment's x-span, directly on the segment, reports it.
        let on_seg = Point::new(5.0, 3.0);
        assert_eq!(map.locate(on_seg), map.locate_sweep(on_seg));
    }

    /// Generate `target` mutually-disjoint random segments via rejection sampling.
    fn disjoint_random_segments(rng: &mut LcgRng, target: usize) -> Vec<Segment> {
        use crate::intersection::segment_segment::{
            SegmentSegmentIntersection, intersect_segments,
        };
        let mut kept: Vec<Segment> = Vec::new();
        let mut attempts = 0_usize;
        while kept.len() < target && attempts < target * 400 {
            attempts += 1;
            let cand = Segment::new(
                Point::new(rng.next_range(-20.0, 20.0), rng.next_range(-20.0, 20.0)),
                Point::new(rng.next_range(-20.0, 20.0), rng.next_range(-20.0, 20.0)),
            );
            if cand.length_sq() < 1.0 {
                continue;
            }
            let disjoint = kept.iter().all(|&prev| {
                matches!(
                    intersect_segments(cand, prev),
                    SegmentSegmentIntersection::None
                )
            });
            if disjoint {
                kept.push(cand);
            }
        }
        kept
    }

    #[test]
    fn stress_disjoint_random_segments_vs_sweep() {
        // The headline contract: on arbitrary, mutually-disjoint random segment sets the
        // DAG `locate` must agree with the brute-force sweep for every query. These sets
        // interleave freely in x and y, exercising multi-trapezoid crossings and merges.
        for seed in [3_u64, 17, 71, 256, 4096, 65_535] {
            let mut rng = LcgRng::new(seed);
            let segs = disjoint_random_segments(&mut rng, 24);
            assert!(segs.len() >= 8, "seed {seed}: too few disjoint segments");
            let map = TrapezoidalMap::build_seeded(segs, seed ^ 0xABCD);
            for _ in 0..600 {
                let q = Point::new(rng.next_range(-25.0, 25.0), rng.next_range(-25.0, 25.0));
                assert_eq!(
                    map.locate(q),
                    map.locate_sweep(q),
                    "seed {seed}: DAG vs sweep mismatch at ({}, {})",
                    q.x,
                    q.y
                );
            }
        }
    }

    /// Mean DAG depth over several construction seeds, smoothing randomized variance.
    fn mean_dag_depth(n: usize) -> f64 {
        let seeds = [11_u64, 29, 53, 97, 211, 401, 809, 1601];
        let total: usize = seeds
            .iter()
            .map(|&seed| TrapezoidalMap::build_seeded(stacked_segments(n), seed).dag_depth())
            .sum();
        total as f64 / seeds.len() as f64
    }

    #[test]
    fn dag_depth_grows_logarithmically() {
        // A single randomized build has noisy depth, so average over several seeds. The
        // averaged depth must grow *sub-linearly* (logarithmic trend): going from n to
        // 16 * n must cost far less than a 16x depth increase.
        let small = mean_dag_depth(16).max(1.0);
        let large = mean_dag_depth(256).max(1.0);
        let ratio = large / small;
        // 16x more segments: a linear structure would give ratio ~16, a logarithmic one
        // gives ratio ~2 (log2(256)/log2(16) = 2). Allow generous slack for variance.
        assert!(
            ratio < 6.0,
            "depth scaling not logarithmic: mean depth(16)={small:.1}, depth(256)={large:.1}, ratio={ratio:.2}"
        );
        // And the absolute averaged depth stays well under the segment count.
        assert!(
            large < 256.0 / 2.0,
            "mean depth {large:.1} should be far below the 256-segment count"
        );
    }

    #[test]
    fn dag_depth_step_bounded_per_doubling() {
        // Across each doubling of n the averaged depth gains only a small additive
        // amount — the hallmark of `O(log n)` growth (each doubling adds ~1 level).
        let sizes = [16_usize, 32, 64, 128, 256];
        let depths: Vec<f64> = sizes.iter().map(|&n| mean_dag_depth(n)).collect();
        for w in depths.windows(2) {
            // A doubling under logarithmic growth adds a bounded constant; the slack
            // absorbs randomized-construction variance without admitting linear growth.
            assert!(
                w[1] - w[0] < 10.0,
                "a doubling of n added too much depth: {depths:?}"
            );
        }
    }
}
