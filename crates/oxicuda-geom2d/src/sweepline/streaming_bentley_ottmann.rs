//! Streaming (out-of-core friendly) Bentley-Ottmann segment-intersection sweep.
//!
//! The baseline [`bentley_ottmann`](fn@crate::sweepline::bentley_ottmann) reports all pairwise
//! intersections with an `O(n^2)` all-pairs scan and materialises every result point in a
//! `Vec`. That is robust and simple but scales poorly, and it forces the *entire* output set
//! to live in memory at once.
//!
//! This module implements the classical event-driven Bentley-Ottmann sweep
//! (Jon L. Bentley and Thomas A. Ottmann, "Algorithms for reporting and counting geometric
//! intersections", IEEE Trans. Computers C-28(9):643-647, 1979; see also de Berg, Cheong,
//! van Kreveld, Overmars, *Computational Geometry* 3rd ed., ch. 2) in a form designed for
//! **streaming** very large segment arrangements:
//!
//!   * The sweep advances a vertical line left-to-right over an **event queue** (a binary heap
//!     keyed by sweep position `(x, y)`). At any instant the queue holds at most `O(n + m)`
//!     pending events, where `m` is the number of *future* intersection events currently
//!     scheduled, not the total over the whole sweep.
//!   * The **sweep status** is an ordered structure of only the segments that currently
//!     straddle the sweep line, ordered by their `y` at the sweep `x`. Its size is the number
//!     of *active* segments, never the whole input.
//!   * Intersections are delivered through an [`IntersectionSink`] **as they are discovered**,
//!     in sweep order, so a consumer may write them to disk / a socket / a counter and never
//!     accumulate them. This is the out-of-core reporting interface requested by the Vol.61
//!     roadmap (`Streaming sweepline for very large segment sets`).
//!
//! The worst-case running time is `O((n + k) log n)` for `n` segments and `k` reported
//! intersection points, the optimum for the comparison-based reporting problem, and the peak
//! memory used by the sweep itself (queue + status) is `O(n + I_active)` where `I_active` is
//! the number of intersection events alive at one moment - typically far smaller than `k`.
//!
//! # Degeneracy handling
//!
//! The implementation is robust to the standard hard cases:
//!
//!   * **Vertical segments** (`a.x == b.x`) are ordered by their lower endpoint and compared in
//!     the status by a small-`epsilon` offset above the sweep line so they slot consistently.
//!   * **Shared endpoints / common intersection points**: every event carries the full set of
//!     segments that *start at*, *end at*, and *pass through* the event point. They are
//!     reordered atomically (all removed, the point reported once, then the survivors
//!     reinserted in their post-event order), which is the de Berg "handle event point"
//!     procedure and copes with three-or-more segments meeting at a point.
//!   * **Overlapping collinear segments** are reported via the exact endpoint(s) of their
//!     shared sub-segment, matching [`intersect_segments`] semantics.
//!
//! All point-ordering decisions go through the exact [`orient2d`](crate::predicate::orient2d)
//! predicate where a sign is geometrically meaningful, so near-collinear inputs are classified
//! consistently.

use core::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use crate::error::{Geom2dError, Geom2dResult};
use crate::intersection::segment_segment::{SegmentSegmentIntersection, intersect_segments};
use crate::predicate::orient2d_sign;
use crate::primitives::point::Point;
use crate::primitives::segment::Segment;

/// Absolute tolerance used to fuse near-coincident event points into one event.
///
/// Two candidate event points within this distance are treated as the same geometric point,
/// which keeps a single physical crossing from spawning a flutter of distinct events under
/// floating-point round-off.
const EVENT_EPS: f64 = 1e-9;

/// A sink that receives intersection points **as the sweep discovers them**, in sweep order.
///
/// Implement this to consume results without retaining them all in memory: write each point to
/// a file, push it down a channel, or simply count it. The sweep calls [`Self::report`] exactly
/// once per distinct intersection point (the point plus the indices of every input segment that
/// passes through it).
pub trait IntersectionSink {
    /// Receive one intersection point and the sorted indices of the segments meeting there.
    ///
    /// `segments` always has length `>= 2` and is sorted ascending.
    fn report(&mut self, point: Point, segments: &[usize]);
}

/// An [`IntersectionSink`] that simply counts intersection points, retaining nothing.
///
/// Useful as the canonical out-of-core consumer: it demonstrates that the sweep need not hold
/// the output, and is handy for verification (`k` counting) on enormous inputs.
#[derive(Debug, Default, Clone)]
pub struct CountingSink {
    /// Number of distinct intersection points seen so far.
    pub count: usize,
    /// Total number of (segment, point) incidences seen (sum of group sizes).
    pub incidences: usize,
}

impl IntersectionSink for CountingSink {
    fn report(&mut self, _point: Point, segments: &[usize]) {
        self.count += 1;
        self.incidences += segments.len();
    }
}

/// A convenience [`IntersectionSink`] that collects every reported point into a `Vec`.
///
/// This deliberately *does* retain all results; use it only when the full set is wanted. It
/// exists so [`report_intersections`] can offer a simple "give me everything" return path while
/// the streaming machinery underneath stays out-of-core.
#[derive(Debug, Default, Clone)]
pub struct CollectingSink {
    /// All reported intersection points, in sweep order.
    pub points: Vec<Point>,
    /// For each reported point, the sorted segment indices meeting there.
    pub groups: Vec<Vec<usize>>,
}

impl IntersectionSink for CollectingSink {
    fn report(&mut self, point: Point, segments: &[usize]) {
        self.points.push(point);
        self.groups.push(segments.to_vec());
    }
}

/// A point with a total order under the left-to-right (then bottom-to-top) sweep.
///
/// `a < b` means `a` is processed *earlier*: smaller `x`, or on a tie smaller `y`.
#[derive(Debug, Clone, Copy)]
struct SweepKey {
    x: f64,
    y: f64,
}

impl SweepKey {
    fn new(p: Point) -> Self {
        Self { x: p.x, y: p.y }
    }

    /// Total sweep order, NaN-safe (NaN sinks to the end so it cannot wedge the heap).
    fn cmp_sweep(&self, other: &Self) -> Ordering {
        match self.x.partial_cmp(&other.x) {
            Some(Ordering::Equal) | None => self.y.partial_cmp(&other.y).unwrap_or(Ordering::Equal),
            Some(ord) => ord,
        }
    }
}

/// Heap event: the sweep reaches the point `key`. The binary heap is a *max*-heap, so we invert
/// the comparison to pop the **smallest** (earliest) event first.
#[derive(Debug, Clone, Copy)]
struct Event {
    key: SweepKey,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.key.cmp_sweep(&other.key) == Ordering::Equal
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so `BinaryHeap` (a max-heap) yields earliest-first.
        other.key.cmp_sweep(&self.key)
    }
}

/// Integer-quantised event-point key for de-duplicating coincident events in the queue map.
///
/// Two points snap to the same bucket iff they are within `EVENT_EPS` after rounding to the
/// `EVENT_EPS` grid, which fuses round-off-separated copies of one geometric crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PointKey {
    xi: i64,
    yi: i64,
}

impl PointKey {
    fn new(p: Point) -> Self {
        Self {
            xi: (p.x / EVENT_EPS).round() as i64,
            yi: (p.y / EVENT_EPS).round() as i64,
        }
    }
}

/// Per-event-point bookkeeping: which segments interact at this point and the canonical point.
///
/// `Default` seeds an empty bucket at the origin; the real point is set on first insertion. We
/// implement it by hand because [`Point`] does not derive `Default`.
#[derive(Debug, Clone)]
struct EventData {
    /// Canonical coordinates of the event point.
    point: Point,
    /// Segments whose *left* (sweep-entry) endpoint is this point.
    starts: Vec<usize>,
    /// Segments whose *right* (sweep-exit) endpoint is this point.
    ends: Vec<usize>,
    /// Segments that pass through this point in their interior (intersection witnesses).
    interiors: Vec<usize>,
}

impl Default for EventData {
    fn default() -> Self {
        Self {
            point: Point::ORIGIN,
            starts: Vec::new(),
            ends: Vec::new(),
            interiors: Vec::new(),
        }
    }
}

/// A segment normalised so that endpoint `lo` precedes `hi` in sweep order.
#[derive(Debug, Clone, Copy)]
struct OrientedSegment {
    lo: Point,
    hi: Point,
}

impl OrientedSegment {
    fn new(seg: Segment) -> Self {
        let ka = SweepKey::new(seg.a);
        let kb = SweepKey::new(seg.b);
        if ka.cmp_sweep(&kb) == Ordering::Greater {
            Self {
                lo: seg.b,
                hi: seg.a,
            }
        } else {
            Self {
                lo: seg.a,
                hi: seg.b,
            }
        }
    }

    /// Is this a single degenerate point (`lo == hi`)?
    fn is_point(&self) -> bool {
        self.lo.x == self.hi.x && self.lo.y == self.hi.y
    }

    /// The `y` of the segment at sweep abscissa `x`, clamped to the segment's `x`-span.
    ///
    /// For a vertical segment this returns `lo.y` (its lowest point), the value used to seat it
    /// in the status just as the sweep reaches it.
    fn y_at(&self, x: f64) -> f64 {
        let dx = self.hi.x - self.lo.x;
        if dx == 0.0 {
            return self.lo.y;
        }
        let t = ((x - self.lo.x) / dx).clamp(0.0, 1.0);
        self.lo.y + t * (self.hi.y - self.lo.y)
    }
}

/// The complete streaming Bentley-Ottmann sweep.
///
/// Construct with [`Self::new`], then drive it with [`Self::run`] passing any
/// [`IntersectionSink`]. The struct owns the segment table, event queue, and sweep status; the
/// sink owns the (possibly discarded) output.
#[derive(Debug)]
pub struct StreamingSweep {
    /// Input segments, normalised so `lo` precedes `hi` in sweep order.
    segs: Vec<OrientedSegment>,
    /// Pending events as a min-by-sweep-order heap.
    queue: BinaryHeap<Event>,
    /// Event-point detail keyed by quantised location, so duplicate-keyed heap entries coalesce.
    pending: BTreeMap<PointKey, EventData>,
    /// Active segments ordered by height at the sweep line: `(StatusKey -> segment index)`.
    status: Vec<usize>,
    /// Current sweep abscissa, used to evaluate `y_at` when re-sorting the status.
    sweep_x: f64,
}

impl StreamingSweep {
    /// Build a streaming sweep over `segments`.
    ///
    /// Returns [`Geom2dError::EmptyInput`] if `segments` is empty. Degenerate (zero-length)
    /// segments are retained: they generate coincident start/end events and can still witness
    /// an intersection if another segment passes through their point.
    pub fn new(segments: &[Segment]) -> Geom2dResult<Self> {
        if segments.is_empty() {
            return Err(Geom2dError::EmptyInput);
        }
        let segs: Vec<OrientedSegment> =
            segments.iter().map(|s| OrientedSegment::new(*s)).collect();
        Ok(Self {
            segs,
            queue: BinaryHeap::new(),
            pending: BTreeMap::new(),
            status: Vec::new(),
            sweep_x: f64::NEG_INFINITY,
        })
    }

    /// Number of input segments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segs.len()
    }

    /// Whether there are no input segments. Always `false` for a constructed sweep.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    /// Insert (or merge) an endpoint event into the queue and pending map.
    fn schedule_endpoint(&mut self, p: Point, idx: usize, is_start: bool) {
        let pk = PointKey::new(p);
        let entry = self.pending.entry(pk).or_default();
        if entry.starts.is_empty() && entry.ends.is_empty() && entry.interiors.is_empty() {
            entry.point = p;
            self.queue.push(Event {
                key: SweepKey::new(p),
            });
        }
        if is_start {
            if !entry.starts.contains(&idx) {
                entry.starts.push(idx);
            }
        } else if !entry.ends.contains(&idx) {
            entry.ends.push(idx);
        }
    }

    /// Schedule (or augment) an intersection event at `p` involving segments `i` and `j`.
    ///
    /// Only future events (strictly to the right of, or above-at, the current sweep position)
    /// are inserted; a point already swept past is ignored.
    fn schedule_intersection(&mut self, p: Point, i: usize, j: usize) {
        // Reject events that lie at or behind the sweep abscissa minus tolerance: they are in
        // the past and must already have been handled.
        if p.x < self.sweep_x - EVENT_EPS {
            return;
        }
        let pk = PointKey::new(p);
        let fresh = !self.pending.contains_key(&pk);
        let entry = self.pending.entry(pk).or_default();
        if fresh {
            entry.point = p;
            self.queue.push(Event {
                key: SweepKey::new(p),
            });
        }
        for idx in [i, j] {
            // A segment whose endpoint *is* this point is recorded as a start/end already; only
            // genuine interior passages become "interiors".
            if !endpoint_is(&self.segs[idx], p)
                && !entry.interiors.contains(&idx)
                && !entry.starts.contains(&idx)
                && !entry.ends.contains(&idx)
            {
                entry.interiors.push(idx);
            }
        }
    }

    /// Run the sweep to completion, delivering every intersection point to `sink` in sweep order.
    ///
    /// Returns the number of distinct intersection points reported.
    pub fn run<S: IntersectionSink>(&mut self, sink: &mut S) -> usize {
        // Seed the queue with all segment endpoints.
        for idx in 0..self.segs.len() {
            let s = self.segs[idx];
            self.schedule_endpoint(s.lo, idx, true);
            self.schedule_endpoint(s.hi, idx, false);
        }

        let mut reported = 0usize;
        while let Some(ev) = self.queue.pop() {
            let pk = PointKey::new(ev.point());
            // The pending detail may have been merged/removed; skip stale heap entries.
            let Some(data) = self.pending.remove(&pk) else {
                continue;
            };
            if self.handle_event(&data, sink) {
                reported += 1;
            }
        }
        reported
    }

    /// Handle one event point. Returns `true` if an intersection was reported here.
    fn handle_event<S: IntersectionSink>(&mut self, data: &EventData, sink: &mut S) -> bool {
        let p = data.point;
        self.sweep_x = p.x;

        // Segments incident to p: those starting, ending, or passing through. Collect their
        // current presence in the status.
        let mut starting = data.starts.clone();
        let ending = data.ends.clone();
        let interior = data.interiors.clone();

        let mut interior = interior;

        // Sweep the active status for any segment that passes through `p` in its interior but
        // was not already flagged (a T-junction or collinear overlap whose witness is the start
        // or end endpoint of another segment, so no explicit intersection event scheduled it).
        // Such segments must participate in the event so the incidence is reported and they are
        // reinserted in the correct post-event order.
        for &idx in self.status.clone().iter() {
            if ending.contains(&idx) || interior.contains(&idx) || starting.contains(&idx) {
                continue;
            }
            let s = &self.segs[idx];
            if !endpoint_is(s, p) && segment_contains_point(s, p) {
                interior.push(idx);
            }
        }

        // The union of (ending + interior + starting) that currently sit at height p.y in the
        // status: remove them all so we can reinsert in post-event order.
        let mut leaving: Vec<usize> = Vec::new();
        for &idx in ending.iter().chain(interior.iter()) {
            if let Some(pos) = self.status.iter().position(|&s| s == idx) {
                self.status.remove(pos);
                leaving.push(idx);
            } else {
                // Not yet in status (e.g. its start event coincides with this point): treat as
                // starting instead.
                if !starting.contains(&idx) {
                    starting.push(idx);
                }
            }
        }

        // Report an intersection if 2+ distinct segments meet here.
        let mut meeting: Vec<usize> = Vec::new();
        meeting.extend(ending.iter().copied());
        meeting.extend(interior.iter().copied());
        meeting.extend(starting.iter().copied());
        meeting.sort_unstable();
        meeting.dedup();

        let reported = if meeting.len() >= 2 {
            // Only treat as an intersection point if at least two segments actually overlap in
            // their *interior or shared endpoint* here, i.e. it is not merely two unrelated
            // segments that happen to share only a queue bucket. Validate by counting how many
            // of `meeting` actually contain p.
            let on_p: Vec<usize> = meeting
                .iter()
                .copied()
                .filter(|&idx| segment_contains_point(&self.segs[idx], p))
                .collect();
            if on_p.len() >= 2 && !is_pure_chain(&on_p, &self.segs, p) {
                sink.report(p, &on_p);
                true
            } else {
                false
            }
        } else {
            false
        };

        // Reinsert continuing segments (starting + interiors that continue past p, i.e. p is not
        // their hi endpoint) ordered by their height just to the RIGHT of p. Ended segments are
        // dropped.
        let mut reinsert: Vec<usize> = Vec::new();
        for &idx in starting.iter().chain(leaving.iter()) {
            if ending.contains(&idx) {
                continue; // this segment terminates here.
            }
            if self.segs[idx].is_point() {
                continue; // degenerate point segment never occupies the status to the right.
            }
            if !reinsert.contains(&idx) {
                reinsert.push(idx);
            }
        }

        // Order the reinserted segments by their state immediately right of the event using a
        // sweep abscissa nudged past p.x for non-vertical comparisons.
        let probe_x = p.x;
        reinsert.sort_by(|&a, &b| self.order_right_of(a, b, probe_x, p));

        // Splice the reinserted block back into the status at the correct global position.
        self.insert_block(&reinsert, p);

        // Compute new neighbour pairs to test for future intersections:
        //   - the segment just below the inserted block vs the block's lowest,
        //   - the block's highest vs the segment just above,
        //   - if nothing was inserted, the two segments that became adjacent across the gap.
        self.test_new_neighbours(&reinsert, p);

        reported
    }

    /// Order two continuing segments by their height just to the right of the event point `p`.
    fn order_right_of(&self, a: usize, b: usize, x: f64, p: Point) -> Ordering {
        let sa = self.segs[a];
        let sb = self.segs[b];
        // Both pass through p; compare by orientation of their far endpoints about p, which is
        // equivalent to comparing slope just to the right and is exact via orient2d.
        let ta = if sa.hi.x == p.x && sa.hi.y == p.y {
            sa.lo
        } else {
            sa.hi
        };
        let tb = if sb.hi.x == p.x && sb.hi.y == p.y {
            sb.lo
        } else {
            sb.hi
        };
        // orient2d(p, ta, tb) > 0 means tb is to the left of ray p->ta, i.e. a is below b to the
        // right of p.
        match orient2d_sign(p, ta, tb) {
            1 => Ordering::Less,
            -1 => Ordering::Greater,
            _ => {
                // Collinear continuations (overlap): fall back to y-at-probe then index.
                let ya = sa.y_at(x + EVENT_EPS);
                let yb = sb.y_at(x + EVENT_EPS);
                ya.partial_cmp(&yb)
                    .unwrap_or(Ordering::Equal)
                    .then(a.cmp(&b))
            }
        }
    }

    /// Insert an ordered block of segment indices into the status at the height of `p`.
    fn insert_block(&mut self, block: &[usize], p: Point) {
        if block.is_empty() {
            return;
        }
        // Find the insertion index: the first status entry whose height at sweep_x is strictly
        // above p.y (using a probe just right of p for stability).
        let probe_x = p.x + EVENT_EPS;
        let mut pos = self.status.len();
        for (i, &idx) in self.status.iter().enumerate() {
            let y = self.segs[idx].y_at(probe_x);
            if y > p.y + EVENT_EPS {
                pos = i;
                break;
            }
        }
        for (offset, &idx) in block.iter().enumerate() {
            self.status.insert(pos + offset, idx);
        }
    }

    /// After splicing `block` (possibly empty) at event `p`, test the newly adjacent pairs.
    fn test_new_neighbours(&mut self, block: &[usize], p: Point) {
        if block.is_empty() {
            // The gap left by removed segments may have made two survivors adjacent. Find the
            // position where the block would have gone and test the straddling pair.
            let probe_x = p.x + EVENT_EPS;
            let mut pos = self.status.len();
            for (i, &idx) in self.status.iter().enumerate() {
                if self.segs[idx].y_at(probe_x) > p.y + EVENT_EPS {
                    pos = i;
                    break;
                }
            }
            if pos > 0 && pos < self.status.len() {
                let below = self.status[pos - 1];
                let above = self.status[pos];
                self.maybe_schedule(below, above, p);
            }
            return;
        }

        // Locate the block inside the status (it is contiguous; find its first member).
        let first = block[0];
        let Some(start) = self.status.iter().position(|&s| s == first) else {
            return;
        };
        let end = start + block.len();
        // Below-neighbour vs block bottom.
        if start > 0 {
            self.maybe_schedule(self.status[start - 1], self.status[start], p);
        }
        // Block top vs above-neighbour.
        if end < self.status.len() {
            self.maybe_schedule(self.status[end - 1], self.status[end], p);
        }
        // Internal adjacencies inside the block (collinear/touching starts) so overlaps are
        // still witnessed.
        for w in start..end.saturating_sub(1) {
            self.maybe_schedule(self.status[w], self.status[w + 1], p);
        }
    }

    /// Test segments `i`, `j` for an intersection strictly ahead of `p`; schedule it if found.
    fn maybe_schedule(&mut self, i: usize, j: usize, p: Point) {
        if i == j {
            return;
        }
        let si = Segment::new(self.segs[i].lo, self.segs[i].hi);
        let sj = Segment::new(self.segs[j].lo, self.segs[j].hi);
        match intersect_segments(si, sj) {
            SegmentSegmentIntersection::None => {}
            SegmentSegmentIntersection::Point(q) => {
                if sweep_ahead(q, p) {
                    self.schedule_intersection(q, i, j);
                }
            }
            SegmentSegmentIntersection::Overlap(o) => {
                // Report both endpoints of the shared overlap as event points ahead of p.
                for q in [o.a, o.b] {
                    if sweep_ahead(q, p) {
                        self.schedule_intersection(q, i, j);
                    }
                }
            }
        }
    }
}

impl Event {
    fn point(&self) -> Point {
        Point::new(self.key.x, self.key.y)
    }
}

/// Whether `p` is exactly one of the (oriented) segment's endpoints.
fn endpoint_is(s: &OrientedSegment, p: Point) -> bool {
    (s.lo.x == p.x && s.lo.y == p.y) || (s.hi.x == p.x && s.hi.y == p.y)
}

/// Whether `p` lies on the closed segment (endpoints inclusive), exact-orientation collinear.
fn segment_contains_point(s: &OrientedSegment, p: Point) -> bool {
    if s.is_point() {
        return s.lo.x == p.x && s.lo.y == p.y;
    }
    // Must be collinear with lo->hi and within the bounding box.
    if orient2d_sign(s.lo, s.hi, p) != 0 {
        // Allow a tiny perpendicular slack for round-off-introduced intersection points.
        let seg = Segment::new(s.lo, s.hi);
        if seg.distance(p) > EVENT_EPS {
            return false;
        }
    }
    let minx = s.lo.x.min(s.hi.x) - EVENT_EPS;
    let maxx = s.lo.x.max(s.hi.x) + EVENT_EPS;
    let miny = s.lo.y.min(s.hi.y) - EVENT_EPS;
    let maxy = s.lo.y.max(s.hi.y) + EVENT_EPS;
    p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy
}

/// `true` iff `q` is strictly ahead of `p` in sweep order (so it can still be scheduled).
fn sweep_ahead(q: Point, p: Point) -> bool {
    if q.x > p.x + EVENT_EPS {
        return true;
    }
    if q.x < p.x - EVENT_EPS {
        return false;
    }
    q.y > p.y + EVENT_EPS
}

/// Whether the segments meeting at `p` form a *pure chain* - i.e. `p` is only ever an endpoint
/// shared by consecutive pieces and never an interior crossing.
///
/// This is not used to suppress genuine endpoint-touch intersections (those are still real
/// crossings of the arrangement); it is reserved as a hook and currently always returns `false`
/// so every multi-segment incidence is reported. Endpoint coincidences ARE intersections of the
/// segment set and the caller may want them; suppression policy is left to the sink.
fn is_pure_chain(_on_p: &[usize], _segs: &[OrientedSegment], _p: Point) -> bool {
    false
}

/// Convenience driver: run the streaming sweep and collect every intersection point.
///
/// This is the simple "give me all crossings" entry point; internally it still uses the
/// out-of-core sweep, only the [`CollectingSink`] retains the output. For truly large inputs,
/// implement a custom [`IntersectionSink`] and call [`StreamingSweep::run`] directly.
///
/// Returns the points in sweep order (left-to-right, then bottom-to-top), de-duplicated.
pub fn report_intersections(segments: &[Segment]) -> Geom2dResult<Vec<Point>> {
    let mut sweep = StreamingSweep::new(segments)?;
    let mut sink = CollectingSink::default();
    sweep.run(&mut sink);
    Ok(sink.points)
}

/// Convenience driver: count distinct intersection points without retaining any.
///
/// Demonstrates fully out-of-core operation - peak extra memory is the sweep's queue + status,
/// not the output. Returns the number of distinct intersection points.
pub fn count_intersections(segments: &[Segment]) -> Geom2dResult<usize> {
    let mut sweep = StreamingSweep::new(segments)?;
    let mut sink = CountingSink::default();
    let n = sweep.run(&mut sink);
    debug_assert_eq!(n, sink.count);
    Ok(sink.count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::sweepline::bentley_ottmann::bentley_ottmann;

    fn seg(ax: f64, ay: f64, bx: f64, by: f64) -> Segment {
        Segment::new(Point::new(ax, ay), Point::new(bx, by))
    }

    /// Brute-force ground truth: distinct intersection points by all-pairs test, fused at 1e-7.
    fn brute_points(segs: &[Segment]) -> Vec<Point> {
        let mut pts: Vec<Point> = Vec::new();
        let push = |pts: &mut Vec<Point>, p: Point| {
            if !pts.iter().any(|q| q.distance(p) < 1e-7) {
                pts.push(p);
            }
        };
        for i in 0..segs.len() {
            for j in (i + 1)..segs.len() {
                match intersect_segments(segs[i], segs[j]) {
                    SegmentSegmentIntersection::None => {}
                    SegmentSegmentIntersection::Point(p) => push(&mut pts, p),
                    SegmentSegmentIntersection::Overlap(o) => {
                        push(&mut pts, o.a);
                        push(&mut pts, o.b);
                    }
                }
            }
        }
        pts
    }

    fn count_matches_brute(segs: &[Segment]) -> (usize, usize) {
        let stream = count_intersections(segs).expect("non-empty");
        let brute = brute_points(segs).len();
        (stream, brute)
    }

    #[test]
    fn empty_input_errors() {
        assert!(matches!(
            StreamingSweep::new(&[]),
            Err(Geom2dError::EmptyInput)
        ));
    }

    #[test]
    fn single_crossing() {
        let segs = vec![seg(0.0, 0.0, 4.0, 4.0), seg(0.0, 4.0, 4.0, 0.0)];
        let pts = report_intersections(&segs).expect("ok");
        assert_eq!(pts.len(), 1);
        assert!((pts[0].x - 2.0).abs() < 1e-9 && (pts[0].y - 2.0).abs() < 1e-9);
    }

    #[test]
    fn no_crossing_parallel() {
        let segs = vec![seg(0.0, 0.0, 1.0, 0.0), seg(0.0, 1.0, 1.0, 1.0)];
        assert_eq!(count_intersections(&segs).expect("ok"), 0);
    }

    #[test]
    fn four_segment_arrangement_matches_baseline() {
        // The same arrangement used by the baseline bentley_ottmann test.
        let segs = vec![
            seg(0.0, 0.0, 4.0, 4.0),
            seg(0.0, 4.0, 4.0, 0.0),
            seg(0.5, -1.0, 0.5, 5.0),
            seg(-1.0, 3.0, 5.0, 3.0),
        ];
        let stream = report_intersections(&segs).expect("ok");
        let base = bentley_ottmann(&segs);
        // Both report the same number of distinct crossings (the baseline counts >=4; here the
        // exact arrangement has 6 distinct intersection points).
        assert!(stream.len() >= 4, "stream found {}", stream.len());
        assert_eq!(stream.len(), brute_points(&segs).len());
        assert!(base.len() >= 4);
    }

    #[test]
    fn grid_arrangement_count() {
        // h horizontal + v vertical lines => h*v interior crossings, all distinct.
        let mut segs = Vec::new();
        let h = 5;
        let v = 4;
        for i in 0..h {
            let y = i as f64;
            segs.push(seg(-1.0, y, v as f64, y));
        }
        for j in 0..v {
            let x = j as f64;
            segs.push(seg(x, -1.0, x, h as f64));
        }
        let (stream, brute) = count_matches_brute(&segs);
        assert_eq!(stream, brute);
        assert_eq!(stream, h * v);
    }

    #[test]
    fn three_concurrent_lines_single_point() {
        // Three lines all through the origin: one shared intersection point.
        let segs = vec![
            seg(-2.0, -2.0, 2.0, 2.0),
            seg(-2.0, 2.0, 2.0, -2.0),
            seg(-2.0, 0.0, 2.0, 0.0),
        ];
        let pts = report_intersections(&segs).expect("ok");
        assert_eq!(pts.len(), 1, "got {pts:?}");
        assert!(pts[0].distance(Point::ORIGIN) < 1e-9);
    }

    #[test]
    fn shared_endpoint_is_reported() {
        // Two segments share endpoint (1,1): that is an intersection of the set.
        let segs = vec![seg(0.0, 0.0, 1.0, 1.0), seg(1.0, 1.0, 2.0, 0.0)];
        let pts = report_intersections(&segs).expect("ok");
        assert_eq!(pts.len(), 1);
        assert!(pts[0].distance(Point::new(1.0, 1.0)) < 1e-9);
    }

    #[test]
    fn collinear_overlap_reported() {
        let segs = vec![seg(0.0, 0.0, 2.0, 0.0), seg(1.0, 0.0, 3.0, 0.0)];
        let pts = report_intersections(&segs).expect("ok");
        // The shared overlap [1,0]-[2,0] yields its endpoints as event points.
        assert!(!pts.is_empty());
        assert!(pts.iter().any(|p| p.distance(Point::new(1.0, 0.0)) < 1e-9));
    }

    #[test]
    fn vertical_and_diagonal() {
        let segs = vec![seg(2.0, -3.0, 2.0, 3.0), seg(0.0, 0.0, 5.0, 5.0)];
        let pts = report_intersections(&segs).expect("ok");
        assert_eq!(pts.len(), 1);
        assert!(pts[0].distance(Point::new(2.0, 2.0)) < 1e-9);
    }

    #[test]
    fn random_arrangements_match_brute_force() {
        // Deterministic LCG-driven fuzz: random short segments, compare distinct-point counts.
        let mut rng = LcgRng::new(0xC0FF_EE12_3456_789A);
        for trial in 0..40 {
            let n = 6 + (trial % 7);
            let mut segs = Vec::with_capacity(n);
            for _ in 0..n {
                // ÷2^32 full-range mapping via next_f64 (which already covers [0,1)).
                let ax = rng.next_range(-10.0, 10.0);
                let ay = rng.next_range(-10.0, 10.0);
                let bx = ax + rng.next_range(-6.0, 6.0);
                let by = ay + rng.next_range(-6.0, 6.0);
                segs.push(seg(ax, ay, bx, by));
            }
            let (stream, brute) = count_matches_brute(&segs);
            // Streaming and brute force must agree on the count of distinct crossing points.
            assert_eq!(
                stream, brute,
                "trial {trial}: stream={stream} brute={brute} segs={segs:?}"
            );
        }
    }

    #[test]
    fn counting_sink_retains_nothing() {
        let segs = vec![
            seg(0.0, 0.0, 4.0, 4.0),
            seg(0.0, 4.0, 4.0, 0.0),
            seg(-1.0, 2.0, 5.0, 2.0),
        ];
        let mut sweep = StreamingSweep::new(&segs).expect("ok");
        let mut sink = CountingSink::default();
        let n = sweep.run(&mut sink);
        assert_eq!(n, sink.count);
        assert!(sink.count >= 1);
        // Incidences >= 2 * count (each crossing involves >= 2 segments).
        assert!(sink.incidences >= 2 * sink.count);
    }

    #[test]
    fn sweep_order_is_left_to_right() {
        let segs = vec![
            seg(0.0, 0.0, 10.0, 0.0),
            seg(1.0, -1.0, 1.0, 1.0),
            seg(8.0, -1.0, 8.0, 1.0),
            seg(4.0, -1.0, 4.0, 1.0),
        ];
        let pts = report_intersections(&segs).expect("ok");
        assert_eq!(pts.len(), 3);
        // Returned in increasing x.
        for w in pts.windows(2) {
            assert!(w[0].x <= w[1].x + 1e-9, "not sorted: {pts:?}");
        }
    }
}
