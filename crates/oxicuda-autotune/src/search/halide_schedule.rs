//! Halide-style schedule-space search for loop-nest optimization.
//!
//! This module implements a miniature version of the schedule search at the
//! heart of the Halide auto-scheduler (Adams et al., 2019).  It models a kernel
//! as an ordered **loop nest** and a **schedule** as an ordered list of
//! legality-checked transforms (split/tile, reorder, vectorize, parallelize,
//! unroll).  A feature-based cost model scores a scheduled nest, and a
//! greedy/beam search explores candidate transforms to find a high-scoring
//! schedule.
//!
//! # Loop-nest IR
//!
//! A [`LoopNest`] is an ordered list of [`Loop`]s, outermost first.  Each loop
//! has a name, an extent (trip count), and tags recording whether it is
//! vectorized, parallelized, or unrolled, plus a flag marking loops produced by
//! tiling.  The product of all extents is the total number of iteration points;
//! this is an *invariant* preserved by every transform (tiling an extent-`N`
//! loop into `ceil(N/t) × t` covers at least `N` points, never fewer).
//!
//! # Schedule primitives
//!
//! | Transform              | Effect                                              |
//! |------------------------|-----------------------------------------------------|
//! | [`Transform::Tile`]    | Split a loop into `outer × inner` by a factor       |
//! | [`Transform::Reorder`] | Permute the loop order                              |
//! | [`Transform::Vectorize`]| Mark the innermost loop vectorized (≤ vector width) |
//! | [`Transform::Parallelize`]| Mark a loop parallel                              |
//! | [`Transform::Unroll`]  | Mark a loop unrolled (records a factor)             |
//!
//! Each transform validates its own legality when applied; illegal transforms
//! (e.g. vectorizing a non-innermost loop, an out-of-range permutation) are
//! rejected with an [`AutotuneError`].
//!
//! # Cost model
//!
//! The default [`FeatureCostModel`] rewards data locality (small working set in
//! the innermost tiles), vector utilization, and parallel speedup, and is
//! defined so that a well-tiled matmul-like nest scores strictly higher than
//! the naive untiled nest.  Higher scores are better.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::search::halide_schedule::{
//!     Loop, LoopNest, ScheduleSearcher, FeatureCostModel,
//! };
//!
//! let nest = LoopNest::new(vec![
//!     Loop::new("i", 1024),
//!     Loop::new("j", 1024),
//!     Loop::new("k", 1024),
//! ]);
//! let searcher = ScheduleSearcher::new(FeatureCostModel::default())
//!     .with_tile_factors(vec![8, 16, 32])
//!     .with_beam_width(4)
//!     .with_max_depth(4);
//! let best = searcher.search(&nest).expect("search yields a schedule");
//! assert!(best.score >= searcher.baseline_score(&nest));
//! ```

use crate::error::AutotuneError;

// ---------------------------------------------------------------------------
// Loop-nest IR
// ---------------------------------------------------------------------------

/// A single loop in the nest, with its name, extent and scheduling tags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loop {
    /// Loop variable name (unique within the nest).
    pub name: String,
    /// Trip count (number of iterations).
    pub extent: u64,
    /// Marked vectorized (only legal on the innermost loop).
    pub vectorized: bool,
    /// Marked parallel.
    pub parallel: bool,
    /// Unroll factor (`1` means not unrolled).
    pub unroll_factor: u64,
    /// True if this loop was produced by tiling (a split product).
    pub tiled: bool,
}

impl Loop {
    /// Creates a plain loop with the given name and extent.
    #[must_use]
    pub fn new(name: impl Into<String>, extent: u64) -> Self {
        Self {
            name: name.into(),
            extent,
            vectorized: false,
            parallel: false,
            unroll_factor: 1,
            tiled: false,
        }
    }
}

/// An ordered loop nest (outermost loop first).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopNest {
    /// The loops, outermost first.
    pub loops: Vec<Loop>,
}

impl LoopNest {
    /// Builds a nest from an ordered list of loops.
    #[must_use]
    pub fn new(loops: Vec<Loop>) -> Self {
        Self { loops }
    }

    /// Number of loops in the nest.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.loops.len()
    }

    /// Product of all loop extents — the total iteration count.
    ///
    /// This is the quantity preserved (up to padding from tiling) by every
    /// transform.
    #[must_use]
    pub fn total_iterations(&self) -> u64 {
        self.loops
            .iter()
            .fold(1_u64, |acc, l| acc.saturating_mul(l.extent))
    }

    /// Index of a loop by name, or `None` if absent.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.loops.iter().position(|l| l.name == name)
    }

    /// The innermost loop (last in order), or `None` if the nest is empty.
    #[must_use]
    pub fn innermost(&self) -> Option<&Loop> {
        self.loops.last()
    }
}

// ---------------------------------------------------------------------------
// Schedule transforms
// ---------------------------------------------------------------------------

/// A single schedule primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transform {
    /// Split the loop named `var` into an outer and inner loop by `factor`.
    ///
    /// The outer loop gets extent `ceil(extent / factor)` and the inner loop
    /// extent `factor`.  `outer × inner ≥ extent`, so all original iterations
    /// are covered (the tail is padded, as in Halide's `split`).
    Tile {
        /// Loop to split.
        var: String,
        /// Inner-loop factor (must be `≥ 1`).
        factor: u64,
    },
    /// Reorder the loops to exactly the given name ordering, which must be a
    /// permutation of the current loop names.
    Reorder {
        /// The desired loop order (a permutation of existing names).
        order: Vec<String>,
    },
    /// Vectorize the innermost loop with the given vector width.  Legal only
    /// when `var` is the innermost loop and its extent is `≤ vector_width`.
    Vectorize {
        /// Loop to vectorize (must be innermost).
        var: String,
        /// Vector lane count.
        vector_width: u64,
    },
    /// Mark the loop named `var` as parallel.
    Parallelize {
        /// Loop to parallelize.
        var: String,
    },
    /// Mark the loop named `var` as unrolled by `factor` (records the factor).
    Unroll {
        /// Loop to unroll.
        var: String,
        /// Unroll factor (`≥ 1`).
        factor: u64,
    },
}

/// Ceiling division `ceil(a / b)` for positive `b`.
fn ceil_div(a: u64, b: u64) -> u64 {
    if b == 0 {
        return a;
    }
    a.div_ceil(b)
}

impl Transform {
    /// Applies this transform to `nest`, returning the transformed nest or an
    /// error describing why the transform is illegal.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] when the transform is illegal
    /// (unknown loop name, zero factor, non-innermost vectorize, vector width
    /// too small, or a non-permutation reorder).
    pub fn apply(&self, nest: &LoopNest) -> Result<LoopNest, AutotuneError> {
        match self {
            Transform::Tile { var, factor } => Self::apply_tile(nest, var, *factor),
            Transform::Reorder { order } => Self::apply_reorder(nest, order),
            Transform::Vectorize { var, vector_width } => {
                Self::apply_vectorize(nest, var, *vector_width)
            }
            Transform::Parallelize { var } => Self::apply_parallelize(nest, var),
            Transform::Unroll { var, factor } => Self::apply_unroll(nest, var, *factor),
        }
    }

    fn apply_tile(nest: &LoopNest, var: &str, factor: u64) -> Result<LoopNest, AutotuneError> {
        if factor == 0 {
            return Err(AutotuneError::BenchmarkFailed(
                "tile factor must be >= 1".to_string(),
            ));
        }
        let idx = nest.index_of(var).ok_or_else(|| {
            AutotuneError::BenchmarkFailed(format!("tile: loop '{var}' not found"))
        })?;
        let original = &nest.loops[idx];
        let outer_extent = ceil_div(original.extent, factor);

        let outer = Loop {
            name: format!("{var}_outer"),
            extent: outer_extent,
            vectorized: false,
            parallel: original.parallel,
            unroll_factor: 1,
            tiled: true,
        };
        let inner = Loop {
            name: format!("{var}_inner"),
            extent: factor,
            vectorized: false,
            parallel: false,
            unroll_factor: 1,
            tiled: true,
        };

        let mut loops = nest.loops.clone();
        // Replace the single loop with [outer, inner] in place.
        loops.splice(idx..=idx, [outer, inner]);
        Ok(LoopNest::new(loops))
    }

    fn apply_reorder(nest: &LoopNest, order: &[String]) -> Result<LoopNest, AutotuneError> {
        if order.len() != nest.loops.len() {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "reorder: expected {} loop names, got {}",
                nest.loops.len(),
                order.len()
            )));
        }
        // Build the reordered list by name lookup, ensuring each requested name
        // exists exactly once (a true permutation).
        let mut used = vec![false; nest.loops.len()];
        let mut reordered = Vec::with_capacity(nest.loops.len());
        for name in order {
            let idx = nest.index_of(name).ok_or_else(|| {
                AutotuneError::BenchmarkFailed(format!("reorder: loop '{name}' not found"))
            })?;
            if used[idx] {
                return Err(AutotuneError::BenchmarkFailed(format!(
                    "reorder: loop '{name}' listed more than once"
                )));
            }
            used[idx] = true;
            reordered.push(nest.loops[idx].clone());
        }
        Ok(LoopNest::new(reordered))
    }

    fn apply_vectorize(
        nest: &LoopNest,
        var: &str,
        vector_width: u64,
    ) -> Result<LoopNest, AutotuneError> {
        if vector_width == 0 {
            return Err(AutotuneError::BenchmarkFailed(
                "vectorize: vector width must be >= 1".to_string(),
            ));
        }
        let idx = nest.index_of(var).ok_or_else(|| {
            AutotuneError::BenchmarkFailed(format!("vectorize: loop '{var}' not found"))
        })?;
        // Legality: vectorization only applies to the innermost (last) loop.
        if idx != nest.loops.len() - 1 {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "vectorize: loop '{var}' is not innermost (index {idx} of {})",
                nest.loops.len()
            )));
        }
        // The innermost extent must fit within the vector width.
        if nest.loops[idx].extent > vector_width {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "vectorize: extent {} exceeds vector width {vector_width}",
                nest.loops[idx].extent
            )));
        }
        let mut loops = nest.loops.clone();
        loops[idx].vectorized = true;
        Ok(LoopNest::new(loops))
    }

    fn apply_parallelize(nest: &LoopNest, var: &str) -> Result<LoopNest, AutotuneError> {
        let idx = nest.index_of(var).ok_or_else(|| {
            AutotuneError::BenchmarkFailed(format!("parallelize: loop '{var}' not found"))
        })?;
        let mut loops = nest.loops.clone();
        loops[idx].parallel = true;
        Ok(LoopNest::new(loops))
    }

    fn apply_unroll(nest: &LoopNest, var: &str, factor: u64) -> Result<LoopNest, AutotuneError> {
        if factor == 0 {
            return Err(AutotuneError::BenchmarkFailed(
                "unroll: factor must be >= 1".to_string(),
            ));
        }
        let idx = nest.index_of(var).ok_or_else(|| {
            AutotuneError::BenchmarkFailed(format!("unroll: loop '{var}' not found"))
        })?;
        let mut loops = nest.loops.clone();
        loops[idx].unroll_factor = factor;
        Ok(LoopNest::new(loops))
    }
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// An ordered list of transforms applied to a base nest, with the resulting
/// scheduled nest and its score.
#[derive(Debug, Clone)]
pub struct Schedule {
    /// The transforms in application order.
    pub transforms: Vec<Transform>,
    /// The nest after all transforms have been applied.
    pub nest: LoopNest,
    /// The cost-model score (higher is better).
    pub score: f64,
}

impl Schedule {
    /// Builds a schedule by applying `transforms` to `base` in order and scoring
    /// the result with `model`.
    ///
    /// # Errors
    ///
    /// Returns the first transform's [`AutotuneError`] if any transform is
    /// illegal.
    pub fn build(
        base: &LoopNest,
        transforms: Vec<Transform>,
        model: &dyn CostModel,
    ) -> Result<Self, AutotuneError> {
        let mut nest = base.clone();
        for t in &transforms {
            nest = t.apply(&nest)?;
        }
        let score = model.score(&nest);
        Ok(Self {
            transforms,
            nest,
            score,
        })
    }
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Scores a scheduled loop nest; higher scores indicate better schedules.
pub trait CostModel {
    /// Returns the score of `nest`.  Larger is better.
    fn score(&self, nest: &LoopNest) -> f64;
}

/// A feature-based heuristic cost model rewarding locality, vectorization and
/// parallelism.
///
/// The score is a weighted sum of three terms:
///
/// 1. **Locality** — the inverse of the *innermost working set*, the product of
///    the innermost loop extents up to (and including) the first tiled loop.
///    Tiling shrinks this working set, so tiled schedules score higher.
/// 2. **Vector utilization** — bonus proportional to the vectorized innermost
///    extent relative to the target width.
/// 3. **Parallel speedup** — bonus proportional to `log2` of the product of
///    parallel-loop extents (Amdahl-style diminishing returns).
#[derive(Debug, Clone)]
pub struct FeatureCostModel {
    /// Cache line / register working-set capacity used to normalize locality.
    pub working_set_capacity: f64,
    /// Weight on the locality reward.
    pub locality_weight: f64,
    /// Weight on the vectorization reward.
    pub vector_weight: f64,
    /// Weight on the parallelism reward.
    pub parallel_weight: f64,
    /// Target vector width for the vectorization reward.
    pub target_vector_width: f64,
}

impl Default for FeatureCostModel {
    fn default() -> Self {
        Self {
            working_set_capacity: 1024.0,
            locality_weight: 1.0,
            vector_weight: 0.25,
            parallel_weight: 0.25,
            target_vector_width: 8.0,
        }
    }
}

impl FeatureCostModel {
    /// The innermost working set: product of extents from the innermost loop
    /// outward, stopping after the first tiled (split) loop is included.  For an
    /// untiled nest this is the full innermost extent; for a tiled nest it is the
    /// (smaller) tile volume.
    fn innermost_working_set(nest: &LoopNest) -> f64 {
        if nest.loops.is_empty() {
            return 1.0;
        }
        let mut ws = 1.0_f64;
        for loop_item in nest.loops.iter().rev() {
            ws *= loop_item.extent.max(1) as f64;
            if loop_item.tiled {
                // Innermost tile boundary reached; the tile volume is the
                // reuse window the cache must hold.
                break;
            }
        }
        ws
    }

    /// Product of extents of all parallel loops (1 if none are parallel).
    fn parallel_volume(nest: &LoopNest) -> f64 {
        nest.loops
            .iter()
            .filter(|l| l.parallel)
            .fold(1.0_f64, |acc, l| acc * l.extent.max(1) as f64)
    }
}

impl CostModel for FeatureCostModel {
    fn score(&self, nest: &LoopNest) -> f64 {
        if nest.loops.is_empty() {
            return 0.0;
        }
        // Locality: reward a small working set relative to capacity. A working
        // set within capacity scores ~1; larger working sets are penalized
        // smoothly toward 0.
        let working_set = Self::innermost_working_set(nest);
        let locality = self.working_set_capacity / (self.working_set_capacity + working_set);

        // Vectorization: reward a vectorized innermost loop near the target
        // width.
        let vector_reward = match nest.innermost() {
            Some(inner) if inner.vectorized => {
                (inner.extent.max(1) as f64 / self.target_vector_width).min(1.0)
            }
            _ => 0.0,
        };

        // Parallelism: log-scaled speedup from parallel loops.
        let parallel_reward = Self::parallel_volume(nest).log2().max(0.0)
            / (1.0 + Self::parallel_volume(nest).log2().max(0.0));

        self.locality_weight * locality
            + self.vector_weight * vector_reward
            + self.parallel_weight * parallel_reward
    }
}

// ---------------------------------------------------------------------------
// Schedule searcher (greedy / beam search)
// ---------------------------------------------------------------------------

/// A greedy/beam searcher over the schedule space for a loop nest.
pub struct ScheduleSearcher<M: CostModel> {
    model: M,
    tile_factors: Vec<u64>,
    vector_width: u64,
    beam_width: usize,
    max_depth: usize,
}

impl<M: CostModel> ScheduleSearcher<M> {
    /// Creates a searcher using `model`, with sensible defaults: tile factors
    /// `[8, 16, 32]`, vector width 8, beam width 4, max depth 4.
    #[must_use]
    pub fn new(model: M) -> Self {
        Self {
            model,
            tile_factors: vec![8, 16, 32],
            vector_width: 8,
            beam_width: 4,
            max_depth: 4,
        }
    }

    /// Sets the candidate tile factors used by generated `Tile` transforms.
    #[must_use]
    pub fn with_tile_factors(mut self, factors: Vec<u64>) -> Self {
        self.tile_factors = factors;
        self
    }

    /// Sets the vector width used by generated `Vectorize` transforms.
    #[must_use]
    pub fn with_vector_width(mut self, width: u64) -> Self {
        self.vector_width = width.max(1);
        self
    }

    /// Sets the beam width (number of partial schedules kept per round).
    #[must_use]
    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width.max(1);
        self
    }

    /// Sets the maximum number of transforms in a schedule.
    #[must_use]
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Scores the unscheduled (naive) nest — the search baseline.
    #[must_use]
    pub fn baseline_score(&self, nest: &LoopNest) -> f64 {
        self.model.score(nest)
    }

    /// Generates the legal candidate transforms applicable to `nest`.
    ///
    /// Produces tile transforms for each non-tiled loop and each tile factor,
    /// parallelization of the outermost loop, and vectorization of the innermost
    /// loop when its extent fits the vector width.
    fn candidate_transforms(&self, nest: &LoopNest) -> Vec<Transform> {
        let mut candidates = Vec::new();

        // Tiling: any not-yet-tiled loop whose extent exceeds the factor.
        for loop_item in &nest.loops {
            if loop_item.tiled {
                continue;
            }
            for &factor in &self.tile_factors {
                if factor >= 2 && loop_item.extent > factor {
                    candidates.push(Transform::Tile {
                        var: loop_item.name.clone(),
                        factor,
                    });
                }
            }
        }

        // Parallelize the outermost loop if not already parallel.
        if let Some(first) = nest.loops.first() {
            if !first.parallel {
                candidates.push(Transform::Parallelize {
                    var: first.name.clone(),
                });
            }
        }

        // Vectorize the innermost loop if it fits and is not yet vectorized.
        if let Some(inner) = nest.innermost() {
            if !inner.vectorized && inner.extent <= self.vector_width {
                candidates.push(Transform::Vectorize {
                    var: inner.name.clone(),
                    vector_width: self.vector_width,
                });
            }
        }

        candidates
    }

    /// Runs beam search and returns the best-scoring legal [`Schedule`] found.
    ///
    /// The empty schedule (the naive nest) is always a candidate, so the result
    /// never scores below [`ScheduleSearcher::baseline_score`].
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] only if the base nest is empty
    /// (no schedule can be formed).
    pub fn search(&self, nest: &LoopNest) -> Result<Schedule, AutotuneError> {
        if nest.loops.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(
                "cannot schedule an empty loop nest".to_string(),
            ));
        }

        // Seed the beam with the empty schedule (naive nest).
        let seed = Schedule::build(nest, Vec::new(), &self.model)?;
        let mut best = seed.clone();
        let mut beam = vec![seed];

        for _depth in 0..self.max_depth {
            let mut next_round: Vec<Schedule> = Vec::new();

            for partial in &beam {
                let candidates = self.candidate_transforms(&partial.nest);
                for transform in candidates {
                    let mut transforms = partial.transforms.clone();
                    transforms.push(transform);
                    // Apply incrementally; skip any that turn out illegal.
                    match Schedule::build(nest, transforms, &self.model) {
                        Ok(child) => {
                            if child.score > best.score {
                                best = child.clone();
                            }
                            next_round.push(child);
                        }
                        Err(_) => continue,
                    }
                }
            }

            if next_round.is_empty() {
                break;
            }

            // Keep the top `beam_width` schedules by score (descending).
            next_round.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            next_round.truncate(self.beam_width);
            beam = next_round;
        }

        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul_nest(n: u64) -> LoopNest {
        LoopNest::new(vec![
            Loop::new("i", n),
            Loop::new("j", n),
            Loop::new("k", n),
        ])
    }

    // (a) Tiling a loop of extent N by factor t yields outer extent ceil(N/t),
    //     inner extent t, with outer * inner >= N.
    #[test]
    fn tile_extents_cover_all_iterations() {
        let nest = LoopNest::new(vec![Loop::new("i", 100)]);
        let t = 32;
        let tiled = Transform::Tile {
            var: "i".to_string(),
            factor: t,
        }
        .apply(&nest)
        .expect("tile legal");

        assert_eq!(tiled.loops.len(), 2);
        let outer = &tiled.loops[0];
        let inner = &tiled.loops[1];
        assert_eq!(outer.extent, ceil_div(100, t)); // ceil(100/32) = 4
        assert_eq!(inner.extent, t); // 32
        assert!(
            outer.extent * inner.extent >= 100,
            "outer*inner must cover N: {}*{} < 100",
            outer.extent,
            inner.extent
        );
        assert!(outer.tiled && inner.tiled);
        assert_eq!(outer.name, "i_outer");
        assert_eq!(inner.name, "i_inner");
    }

    // (b) Total iteration count is invariant (>=) under reorder/tile.
    #[test]
    fn iteration_count_preserved_under_transforms() {
        let nest = matmul_nest(48);
        let base_iters = nest.total_iterations(); // 48^3

        // Reorder preserves the count exactly.
        let reordered = Transform::Reorder {
            order: vec!["k".to_string(), "i".to_string(), "j".to_string()],
        }
        .apply(&nest)
        .expect("reorder legal");
        assert_eq!(reordered.total_iterations(), base_iters);

        // Tiling preserves the count up to padding (never fewer points).
        let tiled = Transform::Tile {
            var: "i".to_string(),
            factor: 16,
        }
        .apply(&nest)
        .expect("tile legal");
        // 48 splits evenly by 16 -> exact.
        assert_eq!(tiled.total_iterations(), base_iters);

        // A non-dividing factor pads upward but never drops points.
        let tiled_pad = Transform::Tile {
            var: "i".to_string(),
            factor: 7,
        }
        .apply(&nest)
        .expect("tile legal");
        assert!(
            tiled_pad.total_iterations() >= base_iters,
            "padded tiling must visit >= N points: {} < {}",
            tiled_pad.total_iterations(),
            base_iters
        );
        // ceil(48/7) = 7, so 7*7 = 49 in the i dimension.
        assert_eq!(tiled_pad.total_iterations(), 49 * 48 * 48);
    }

    // (c) Reorder permutes loop order as specified and rejects illegal perms.
    #[test]
    fn reorder_permutes_and_rejects_illegal() {
        let nest = matmul_nest(8);
        let reordered = Transform::Reorder {
            order: vec!["j".to_string(), "k".to_string(), "i".to_string()],
        }
        .apply(&nest)
        .expect("legal permutation");
        let names: Vec<&str> = reordered.loops.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["j", "k", "i"]);

        // Wrong length.
        assert!(
            Transform::Reorder {
                order: vec!["i".to_string(), "j".to_string()],
            }
            .apply(&nest)
            .is_err()
        );
        // Duplicate name (not a permutation).
        assert!(
            Transform::Reorder {
                order: vec!["i".to_string(), "i".to_string(), "j".to_string()],
            }
            .apply(&nest)
            .is_err()
        );
        // Unknown name.
        assert!(
            Transform::Reorder {
                order: vec!["i".to_string(), "j".to_string(), "z".to_string()],
            }
            .apply(&nest)
            .is_err()
        );
    }

    // (d) Vectorize is only legal on the innermost loop; rejects non-innermost.
    #[test]
    fn vectorize_only_innermost() {
        // Innermost loop "k" with small extent fits the vector width.
        let nest = LoopNest::new(vec![
            Loop::new("i", 64),
            Loop::new("j", 64),
            Loop::new("k", 8),
        ]);
        let ok = Transform::Vectorize {
            var: "k".to_string(),
            vector_width: 8,
        }
        .apply(&nest)
        .expect("innermost vectorize legal");
        assert!(ok.loops[2].vectorized);

        // Vectorizing a non-innermost loop must be rejected.
        assert!(
            Transform::Vectorize {
                var: "i".to_string(),
                vector_width: 8,
            }
            .apply(&nest)
            .is_err(),
            "vectorizing a non-innermost loop must be illegal"
        );
        // Extent exceeding the vector width is rejected.
        assert!(
            Transform::Vectorize {
                var: "j".to_string(),
                vector_width: 8,
            }
            .apply(&nest)
            .is_err()
        );
    }

    // (e) On a large matmul-like nest, a tiled schedule scores STRICTLY better
    //     than the naive untiled one (locality reward).
    #[test]
    fn tiled_beats_naive() {
        let model = FeatureCostModel::default();
        let nest = matmul_nest(1024);

        let naive_score = model.score(&nest);

        // Tile the innermost loop to shrink the working set.
        let tiled = Transform::Tile {
            var: "k".to_string(),
            factor: 16,
        }
        .apply(&nest)
        .expect("tile legal");
        let tiled_score = model.score(&tiled);

        assert!(
            tiled_score > naive_score,
            "tiled schedule must score strictly better: tiled={tiled_score}, naive={naive_score}"
        );
    }

    // (f) Beam search returns a legal schedule whose score >= naive baseline.
    #[test]
    fn beam_search_beats_or_matches_baseline() {
        let model = FeatureCostModel::default();
        let nest = matmul_nest(1024);
        let searcher = ScheduleSearcher::new(model)
            .with_tile_factors(vec![8, 16, 32])
            .with_beam_width(4)
            .with_max_depth(4);

        let baseline = searcher.baseline_score(&nest);
        let best = searcher.search(&nest).expect("search yields schedule");

        assert!(
            best.score >= baseline,
            "beam-search result must be >= baseline: best={}, baseline={baseline}",
            best.score
        );

        // The returned schedule must be legal: re-applying its transforms to the
        // base nest reproduces its nest without error.
        let replay = Schedule::build(&nest, best.transforms.clone(), &FeatureCostModel::default())
            .expect("returned schedule must be legal");
        assert_eq!(replay.nest, best.nest);

        // On a large nest the search should actually improve via tiling.
        assert!(
            best.score > baseline,
            "search should improve a large naive nest via tiling"
        );
    }

    #[test]
    fn search_rejects_empty_nest() {
        let searcher = ScheduleSearcher::new(FeatureCostModel::default());
        let empty = LoopNest::new(vec![]);
        assert!(searcher.search(&empty).is_err());
    }

    #[test]
    fn tile_rejects_zero_factor_and_unknown_loop() {
        let nest = matmul_nest(64);
        assert!(
            Transform::Tile {
                var: "i".to_string(),
                factor: 0,
            }
            .apply(&nest)
            .is_err()
        );
        assert!(
            Transform::Tile {
                var: "zzz".to_string(),
                factor: 8,
            }
            .apply(&nest)
            .is_err()
        );
    }

    #[test]
    fn parallelize_and_unroll_tag_loops() {
        let nest = matmul_nest(64);
        let par = Transform::Parallelize {
            var: "i".to_string(),
        }
        .apply(&nest)
        .expect("parallelize legal");
        assert!(par.loops[0].parallel);

        let unr = Transform::Unroll {
            var: "k".to_string(),
            factor: 4,
        }
        .apply(&nest)
        .expect("unroll legal");
        assert_eq!(unr.loops[2].unroll_factor, 4);

        assert!(
            Transform::Unroll {
                var: "k".to_string(),
                factor: 0,
            }
            .apply(&nest)
            .is_err()
        );
    }

    #[test]
    fn schedule_build_applies_in_order() {
        let model = FeatureCostModel::default();
        let nest = matmul_nest(256);
        let schedule = Schedule::build(
            &nest,
            vec![
                Transform::Tile {
                    var: "k".to_string(),
                    factor: 16,
                },
                Transform::Parallelize {
                    var: "i".to_string(),
                },
            ],
            &model,
        )
        .expect("schedule legal");
        // After tiling k, the nest has 4 loops; i is still outermost and parallel.
        assert_eq!(schedule.nest.loops.len(), 4);
        assert!(schedule.nest.loops[0].parallel);
        assert!(schedule.score > 0.0);
    }
}
