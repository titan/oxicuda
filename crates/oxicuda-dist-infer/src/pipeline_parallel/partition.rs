//! Layer → pipeline-stage partition planning.
//!
//! Given `n_layers` transformer blocks and `n_stages` pipeline ranks, decide
//! which contiguous block of layers each stage owns. Two planners are provided:
//!
//! * **Balanced** — distribute layers as evenly as possible; the first
//!   `n_layers % n_stages` stages receive one extra layer. This is the standard
//!   Megatron `--num-layers-per-virtual-pipeline-stage` default.
//! * **Memory-aware** — given a per-layer cost (e.g. parameter bytes or FLOPs),
//!   greedily grow each stage until its cumulative cost crosses an equal share,
//!   so heterogeneous layers (e.g. an embedding stage) don't overload one rank.
//!
//! Both produce a `LayerPartition`: a list of half-open `[start, end)` ranges,
//! one per stage, that exactly tile `0 .. n_layers` with no gaps or overlaps.

use crate::error::{DistInferError, DistInferResult};

// ─── StageRange ────────────────────────────────────────────────────────────────

/// The contiguous half-open range of layers `[start, end)` owned by one stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageRange {
    /// Pipeline-stage (rank) index.
    pub stage: usize,
    /// First layer owned by this stage (inclusive).
    pub start: usize,
    /// One past the last layer owned by this stage (exclusive).
    pub end: usize,
}

impl StageRange {
    /// Number of layers in this stage.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the stage owns no layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Whether `layer` belongs to this stage.
    #[must_use]
    pub fn contains(&self, layer: usize) -> bool {
        layer >= self.start && layer < self.end
    }
}

// ─── LayerPartition ────────────────────────────────────────────────────────────

/// A complete partition of `n_layers` across `n_stages` pipeline ranks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPartition {
    /// One range per stage, in stage order.
    ranges: Vec<StageRange>,
    /// Total layers partitioned.
    n_layers: usize,
}

impl LayerPartition {
    /// Balanced layer partition: each stage gets `⌊L/P⌋` layers; the first
    /// `L mod P` stages get one extra so the assignment is as even as possible
    /// and contiguous.
    ///
    /// # Errors
    ///
    /// * [`DistInferError::TooFewRanks`] if `n_stages == 0`.
    /// * [`DistInferError::DimensionMismatch`] if `n_layers < n_stages` (a
    ///   pipeline cannot have an empty stage).
    pub fn balanced(n_layers: usize, n_stages: usize) -> DistInferResult<Self> {
        if n_stages == 0 {
            return Err(DistInferError::TooFewRanks {
                needed: 1,
                world_size: 0,
            });
        }
        if n_layers < n_stages {
            return Err(DistInferError::DimensionMismatch {
                expected: n_stages,
                got: n_layers,
            });
        }
        let base = n_layers / n_stages;
        let extra = n_layers % n_stages;
        let mut ranges = Vec::with_capacity(n_stages);
        let mut cursor = 0usize;
        for stage in 0..n_stages {
            let len = base + usize::from(stage < extra);
            ranges.push(StageRange {
                stage,
                start: cursor,
                end: cursor + len,
            });
            cursor += len;
        }
        debug_assert_eq!(cursor, n_layers);
        Ok(Self { ranges, n_layers })
    }

    /// Memory- / cost-aware partition.
    ///
    /// `costs[l]` is the relative cost of layer `l` (parameter bytes, FLOPs,
    /// activation memory — any positive weight). Layers are assigned to stages
    /// greedily so each stage's cumulative cost stays close to the equal share
    /// `total / n_stages`, while remaining **contiguous**. This keeps the most
    /// expensive stage's load minimal among contiguous partitions of this
    /// greedy family.
    ///
    /// # Errors
    ///
    /// * [`DistInferError::TooFewRanks`] if `n_stages == 0`.
    /// * [`DistInferError::DimensionMismatch`] if `costs` is empty or shorter
    ///   than `n_stages`.
    pub fn cost_aware(costs: &[f32], n_stages: usize) -> DistInferResult<Self> {
        if n_stages == 0 {
            return Err(DistInferError::TooFewRanks {
                needed: 1,
                world_size: 0,
            });
        }
        let n_layers = costs.len();
        if n_layers < n_stages {
            return Err(DistInferError::DimensionMismatch {
                expected: n_stages,
                got: n_layers,
            });
        }
        let total: f32 = costs.iter().copied().map(f32::abs).sum();
        let share = total / n_stages as f32;

        let mut ranges = Vec::with_capacity(n_stages);
        let mut cursor = 0usize;
        let mut acc = 0.0_f32;
        for stage in 0..n_stages {
            let start = cursor;
            // Reserve at least one layer for every remaining stage.
            let stages_left_after = n_stages - stage - 1;
            let max_end = n_layers - stages_left_after;
            // Grow until the cumulative target for this stage is reached, but
            // always take ≥ 1 layer and never starve later stages.
            let target = share * (stage + 1) as f32;
            // Always consume at least the first layer of this stage.
            acc += costs[cursor].abs();
            cursor += 1;
            while cursor < max_end {
                let next_cost = costs[cursor].abs();
                // Stop if adding the next layer overshoots the target more than
                // stopping here undershoots it (classic balanced-greedy split).
                let over = (acc + next_cost - target).abs();
                let under = (target - acc).abs();
                if over >= under {
                    break;
                }
                acc += next_cost;
                cursor += 1;
            }
            ranges.push(StageRange {
                stage,
                start,
                end: cursor,
            });
        }
        // The last stage must consume any remaining layers.
        if let Some(last) = ranges.last_mut() {
            last.end = n_layers;
        }
        debug_assert_eq!(ranges.last().map(|r| r.end), Some(n_layers));
        Ok(Self { ranges, n_layers })
    }

    /// Per-stage ranges in stage order.
    #[must_use]
    pub fn ranges(&self) -> &[StageRange] {
        &self.ranges
    }

    /// Number of stages.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.ranges.len()
    }

    /// Total layers partitioned.
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Which stage owns `layer`, if any.
    #[must_use]
    pub fn stage_of(&self, layer: usize) -> Option<usize> {
        self.ranges
            .iter()
            .find(|r| r.contains(layer))
            .map(|r| r.stage)
    }

    /// Verify the partition exactly tiles `0..n_layers` with no gap/overlap and
    /// no empty stage.
    #[must_use]
    pub fn is_valid_tiling(&self) -> bool {
        if self.ranges.is_empty() {
            return false;
        }
        let mut expected = 0usize;
        for (i, r) in self.ranges.iter().enumerate() {
            if r.stage != i || r.start != expected || r.is_empty() {
                return false;
            }
            expected = r.end;
        }
        expected == self.n_layers
    }

    /// The maximum per-stage cost given `costs` (the pipeline's critical path
    /// per micro-batch under this partition). Returns `0.0` for an empty
    /// partition.
    #[must_use]
    pub fn max_stage_cost(&self, costs: &[f32]) -> f32 {
        self.ranges
            .iter()
            .map(|r| {
                costs[r.start..r.end]
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .sum::<f32>()
            })
            .fold(0.0_f32, f32::max)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_even_split() {
        let p = LayerPartition::balanced(12, 4).expect("partition");
        assert_eq!(p.n_stages(), 4);
        for r in p.ranges() {
            assert_eq!(r.len(), 3);
        }
        assert!(p.is_valid_tiling());
    }

    #[test]
    fn balanced_remainder_front_loaded() {
        // 10 layers, 4 stages → [3,3,2,2].
        let p = LayerPartition::balanced(10, 4).expect("partition");
        let lens: Vec<usize> = p.ranges().iter().map(StageRange::len).collect();
        assert_eq!(lens, vec![3, 3, 2, 2]);
        assert!(p.is_valid_tiling());
    }

    #[test]
    fn balanced_tiles_exactly() {
        for (l, s) in [(7, 3), (16, 5), (32, 8), (5, 5), (100, 7)] {
            let p = LayerPartition::balanced(l, s).expect("partition");
            assert!(p.is_valid_tiling(), "L={l} S={s} not a valid tiling");
            assert_eq!(p.n_layers(), l);
        }
    }

    #[test]
    fn balanced_stage_of_lookup() {
        let p = LayerPartition::balanced(10, 4).expect("partition"); // [0,3)[3,6)[6,8)[8,10)
        assert_eq!(p.stage_of(0), Some(0));
        assert_eq!(p.stage_of(2), Some(0));
        assert_eq!(p.stage_of(3), Some(1));
        assert_eq!(p.stage_of(7), Some(2));
        assert_eq!(p.stage_of(9), Some(3));
        assert_eq!(p.stage_of(10), None);
    }

    #[test]
    fn balanced_too_few_layers_errors() {
        assert!(matches!(
            LayerPartition::balanced(3, 4),
            Err(DistInferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn balanced_zero_stages_errors() {
        assert!(matches!(
            LayerPartition::balanced(4, 0),
            Err(DistInferError::TooFewRanks { .. })
        ));
    }

    #[test]
    fn cost_aware_uniform_matches_balanced() {
        let costs = vec![1.0_f32; 12];
        let p = LayerPartition::cost_aware(&costs, 4).expect("partition");
        assert!(p.is_valid_tiling());
        // Uniform costs → 3 layers each.
        for r in p.ranges() {
            assert_eq!(r.len(), 3, "uniform cost should split evenly");
        }
    }

    #[test]
    fn cost_aware_heavy_first_layer_isolates_it() {
        // Layer 0 is 10× heavier; with 2 stages it should mostly stand alone.
        let mut costs = vec![1.0_f32; 6];
        costs[0] = 10.0;
        let p = LayerPartition::cost_aware(&costs, 2).expect("partition");
        assert!(p.is_valid_tiling());
        // The expensive layer 0 should be in stage 0, and stage 0 should be
        // small (just the heavy layer) so stage 1 absorbs the rest.
        assert_eq!(p.stage_of(0), Some(0));
        assert_eq!(p.ranges()[0].len(), 1, "heavy layer should be isolated");
        // Max-stage cost should beat the naive equal-split [0..3)=12 vs 11.
        let naive = LayerPartition::balanced(6, 2).expect("balanced");
        assert!(
            p.max_stage_cost(&costs) <= naive.max_stage_cost(&costs),
            "cost-aware must not be worse than balanced on the bottleneck"
        );
    }

    #[test]
    fn cost_aware_tiles_exactly() {
        let costs: Vec<f32> = (0..20).map(|i| (i % 5 + 1) as f32).collect();
        let p = LayerPartition::cost_aware(&costs, 6).expect("partition");
        assert!(p.is_valid_tiling());
        assert_eq!(p.n_layers(), 20);
    }

    #[test]
    fn cost_aware_too_few_layers_errors() {
        assert!(matches!(
            LayerPartition::cost_aware(&[1.0, 2.0], 4),
            Err(DistInferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn stage_range_helpers() {
        let r = StageRange {
            stage: 1,
            start: 3,
            end: 6,
        };
        assert_eq!(r.len(), 3);
        assert!(!r.is_empty());
        assert!(r.contains(3));
        assert!(r.contains(5));
        assert!(!r.contains(6));
        assert!(!r.contains(2));
    }
}
