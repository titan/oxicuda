//! Once-For-All (OFA) elastic supernet with progressive shrinking.
//!
//! Reference: Cai, Gan, Wang, Zhang & Han, "Once-for-All: Train One Network and
//! Specialize it for Efficient Deployment", ICLR 2020.
//!
//! OFA trains a single super-network whose sub-networks share weights and can be
//! sliced along **three elastic axes** without retraining:
//!
//! - **elastic depth**  — each of the `n_units` sequential units keeps its first
//!   `d ∈ depth_choices` blocks (the rest are skipped, identity-forwarded);
//! - **elastic width**  — each kept block uses one of `width_choices` channel
//!   expansion ratios;
//! - **elastic kernel** — each kept block uses one of `kernel_choices` kernel
//!   sizes (a larger kernel's weights contain the smaller kernel as a centred
//!   sub-tensor — "kernel transform" — so any kernel is sliceable).
//!
//! Weight sharing means a smaller kernel / fewer channels / shallower depth are
//! always **prefixes / centred sub-tensors** of the largest configuration, which
//! is exactly what makes a sampled subnet a *no-cost slice*.
//!
//! ## Progressive shrinking
//!
//! Training does not expose all subnets at once; it follows a schedule
//! ([`ShrinkPhase`]) that first trains the full network, then progressively
//! allows smaller kernels, then smaller depths, then smaller widths. Each phase
//! widens the set of admissible choices on one axis. [`ShrinkSchedule`]
//! reproduces the canonical OFA stage order and exposes, for any training step,
//! the currently-admissible choices on each axis so a sampler only draws subnets
//! the supernet has been prepared to serve.
//!
//! This module models the *configuration / sampling / cost* layer (the part that
//! is CPU-deterministic and benchmark-relevant); the actual elastic convolution
//! forward pass delegates to [`crate::ops::mbconv_ops`] for MAC / parameter
//! accounting via [`OfaBlockConfig::to_mbconv`].

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;
use crate::ops::mbconv_ops::{MbConvSpec, mbconv_mac_count, mbconv_param_count};

// ─── OfaBlockConfig ─────────────────────────────────────────────────────────────

/// One resolved elastic block: a kernel size and an expansion (width) ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfaBlockConfig {
    /// Depthwise kernel size (square).
    pub kernel: usize,
    /// Channel expansion ratio used in the inverted residual.
    pub expand_ratio: usize,
}

impl OfaBlockConfig {
    /// Convert to an [`MbConvSpec`] for cost accounting, given the unit's input
    /// / output channels and stride.
    #[must_use]
    pub fn to_mbconv(&self, in_ch: usize, out_ch: usize, stride: usize) -> MbConvSpec {
        MbConvSpec {
            in_ch,
            out_ch,
            stride,
            expand_ratio: self.expand_ratio,
            kernel: self.kernel,
        }
    }
}

// ─── OfaUnit ───────────────────────────────────────────────────────────────────

/// One elastic unit (a stage of identical-resolution blocks).
///
/// The unit reserves `max_depth` block slots; an active subnet uses the first
/// `active_depth` of them. The first block of a unit applies the unit `stride`
/// (down-sampling); the remaining blocks use stride 1 and `out_ch` channels in
/// and out (standard MobileNet/OFA stage layout).
#[derive(Debug, Clone)]
pub struct OfaUnit {
    /// Input channels to the unit's first block.
    pub in_ch: usize,
    /// Output channels (also the in/out of every non-first block).
    pub out_ch: usize,
    /// Stride applied by the unit's first block (1 or 2).
    pub stride: usize,
    /// Maximum number of blocks reserved in this unit.
    pub max_depth: usize,
    /// Currently active number of blocks (`1..=max_depth`).
    pub active_depth: usize,
    /// Per-active-block configurations (`len() == active_depth`).
    pub blocks: Vec<OfaBlockConfig>,
}

impl OfaUnit {
    /// MAC count of the active portion of this unit over a `h × w` input.
    #[must_use]
    pub fn mac_count(&self, h: usize, w: usize) -> u64 {
        let mut total = 0u64;
        let mut cur_h = h;
        let mut cur_w = w;
        for (i, blk) in self.blocks.iter().enumerate() {
            let (cin, cout, stride) = if i == 0 {
                (self.in_ch, self.out_ch, self.stride)
            } else {
                (self.out_ch, self.out_ch, 1)
            };
            let spec = blk.to_mbconv(cin, cout, stride);
            total = total.saturating_add(mbconv_mac_count(&spec, cur_h, cur_w));
            let s = stride.max(1);
            cur_h /= s;
            cur_w /= s;
        }
        total
    }

    /// Parameter count of the active portion of this unit.
    #[must_use]
    pub fn param_count(&self) -> u64 {
        let mut total = 0u64;
        for (i, blk) in self.blocks.iter().enumerate() {
            let (cin, cout, stride) = if i == 0 {
                (self.in_ch, self.out_ch, self.stride)
            } else {
                (self.out_ch, self.out_ch, 1)
            };
            let spec = blk.to_mbconv(cin, cout, stride);
            total = total.saturating_add(mbconv_param_count(&spec));
        }
        total
    }
}

// ─── ShrinkPhase ───────────────────────────────────────────────────────────────

/// A phase of OFA progressive shrinking. Each phase relaxes one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShrinkPhase {
    /// Train only the full (largest) network: every axis pinned to its max.
    FullNetwork,
    /// Allow smaller kernels (depth + width still pinned to max).
    ElasticKernel,
    /// Allow smaller depths as well (width still pinned to max).
    ElasticDepth,
    /// Allow smaller widths as well — the full elastic space.
    ElasticWidth,
}

impl ShrinkPhase {
    /// Canonical OFA phase order.
    #[must_use]
    pub fn order() -> &'static [ShrinkPhase] {
        &[
            ShrinkPhase::FullNetwork,
            ShrinkPhase::ElasticKernel,
            ShrinkPhase::ElasticDepth,
            ShrinkPhase::ElasticWidth,
        ]
    }
}

// ─── ShrinkSchedule ─────────────────────────────────────────────────────────────

/// Progressive-shrinking schedule: maps a global training step to the active
/// [`ShrinkPhase`] given equal-length phase boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShrinkSchedule {
    /// Number of training steps each phase runs for.
    pub steps_per_phase: usize,
}

impl ShrinkSchedule {
    /// Construct a schedule.
    ///
    /// # Errors
    /// [`NasError::InvalidNumOps`] if `steps_per_phase == 0`.
    pub fn new(steps_per_phase: usize) -> NasResult<Self> {
        if steps_per_phase == 0 {
            return Err(NasError::InvalidNumOps);
        }
        Ok(Self { steps_per_phase })
    }

    /// The active phase at training `step`. Steps beyond the last boundary stay
    /// in the final [`ShrinkPhase::ElasticWidth`] phase.
    #[must_use]
    pub fn phase_at(&self, step: usize) -> ShrinkPhase {
        let idx = (step / self.steps_per_phase).min(ShrinkPhase::order().len() - 1);
        ShrinkPhase::order()[idx]
    }

    /// Total number of steps for all four phases.
    #[must_use]
    pub fn total_steps(&self) -> usize {
        self.steps_per_phase * ShrinkPhase::order().len()
    }
}

// ─── OfaSpace ──────────────────────────────────────────────────────────────────

/// The Once-For-All search space: per-unit stage layout plus the three elastic
/// axes' candidate sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfaSpace {
    /// `(in_ch, out_ch, stride, max_depth)` for each sequential unit.
    pub stages: Vec<(usize, usize, usize, usize)>,
    /// Admissible kernel sizes, ascending (e.g. `[3, 5, 7]`).
    pub kernel_choices: Vec<usize>,
    /// Admissible expansion ratios, ascending (e.g. `[3, 4, 6]`).
    pub width_choices: Vec<usize>,
    /// Admissible per-unit depths, ascending (e.g. `[2, 3, 4]`).
    pub depth_choices: Vec<usize>,
    /// Input resolution `(h, w)`.
    pub resolution: (usize, usize),
}

impl OfaSpace {
    /// Construct and validate an OFA space.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if any axis or `stages` is empty, or the
    ///   resolution has a zero dimension.
    /// - [`NasError::InvalidNumOps`] if any candidate / stage field that must be
    ///   positive is `0`.
    /// - [`NasError::InvalidArchEncoding`] if a stage's `max_depth` is smaller
    ///   than the largest `depth_choice` (it could never be served).
    pub fn new(
        stages: Vec<(usize, usize, usize, usize)>,
        kernel_choices: Vec<usize>,
        width_choices: Vec<usize>,
        depth_choices: Vec<usize>,
        resolution: (usize, usize),
    ) -> NasResult<Self> {
        if stages.is_empty()
            || kernel_choices.is_empty()
            || width_choices.is_empty()
            || depth_choices.is_empty()
            || resolution.0 == 0
            || resolution.1 == 0
        {
            return Err(NasError::EmptySearchSpace);
        }
        for &v in kernel_choices
            .iter()
            .chain(&width_choices)
            .chain(&depth_choices)
        {
            if v == 0 {
                return Err(NasError::InvalidNumOps);
            }
        }
        let max_depth_choice = depth_choices.iter().copied().max().unwrap_or(0);
        for &(cin, cout, stride, max_depth) in &stages {
            if cin == 0 || cout == 0 || stride == 0 || max_depth == 0 {
                return Err(NasError::InvalidNumOps);
            }
            if max_depth < max_depth_choice {
                return Err(NasError::InvalidArchEncoding);
            }
        }
        Ok(Self {
            stages,
            kernel_choices,
            width_choices,
            depth_choices,
            resolution,
        })
    }

    /// Admissible choices on each axis under the given shrink phase.
    ///
    /// Returns `(kernels, widths, depths)`. Pinned axes collapse to a single
    /// element — the maximum value — exactly matching OFA progressive shrinking.
    #[must_use]
    pub fn admissible(&self, phase: ShrinkPhase) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let max_of = |xs: &[usize]| vec![xs.iter().copied().max().unwrap_or(0)];
        match phase {
            ShrinkPhase::FullNetwork => (
                max_of(&self.kernel_choices),
                max_of(&self.width_choices),
                max_of(&self.depth_choices),
            ),
            ShrinkPhase::ElasticKernel => (
                self.kernel_choices.clone(),
                max_of(&self.width_choices),
                max_of(&self.depth_choices),
            ),
            ShrinkPhase::ElasticDepth => (
                self.kernel_choices.clone(),
                max_of(&self.width_choices),
                self.depth_choices.clone(),
            ),
            ShrinkPhase::ElasticWidth => (
                self.kernel_choices.clone(),
                self.width_choices.clone(),
                self.depth_choices.clone(),
            ),
        }
    }

    /// Sample a subnet whose choices respect the given shrink `phase`.
    ///
    /// Each unit independently draws an active depth from the admissible depths,
    /// then each active block draws an admissible kernel and width.
    ///
    /// # Errors
    /// Propagates construction errors (cannot fire for a validated space).
    pub fn sample(&self, phase: ShrinkPhase, rng: &mut LcgRng) -> NasResult<OfaSubnet> {
        let (kernels, widths, depths) = self.admissible(phase);
        let mut units = Vec::with_capacity(self.stages.len());
        for &(cin, cout, stride, max_depth) in &self.stages {
            let d = depths[rng.next_usize(depths.len())].min(max_depth);
            let mut blocks = Vec::with_capacity(d);
            for _ in 0..d {
                let kernel = kernels[rng.next_usize(kernels.len())];
                let expand_ratio = widths[rng.next_usize(widths.len())];
                blocks.push(OfaBlockConfig {
                    kernel,
                    expand_ratio,
                });
            }
            units.push(OfaUnit {
                in_ch: cin,
                out_ch: cout,
                stride,
                max_depth,
                active_depth: d,
                blocks,
            });
        }
        OfaSubnet::new(units, self.resolution)
    }

    /// The maximal ("teacher") subnet: every axis at its largest value.
    ///
    /// # Errors
    /// Propagates construction errors.
    pub fn max_subnet(&self) -> NasResult<OfaSubnet> {
        self.extreme_subnet(true)
    }

    /// The minimal subnet: every axis at its smallest value.
    ///
    /// # Errors
    /// Propagates construction errors.
    pub fn min_subnet(&self) -> NasResult<OfaSubnet> {
        self.extreme_subnet(false)
    }

    fn extreme_subnet(&self, maximal: bool) -> NasResult<OfaSubnet> {
        let pick = |xs: &[usize]| -> usize {
            if maximal {
                xs.iter().copied().max().unwrap_or(0)
            } else {
                xs.iter().copied().min().unwrap_or(0)
            }
        };
        let kernel = pick(&self.kernel_choices);
        let expand_ratio = pick(&self.width_choices);
        let depth = pick(&self.depth_choices);
        let mut units = Vec::with_capacity(self.stages.len());
        for &(cin, cout, stride, max_depth) in &self.stages {
            let d = depth.min(max_depth);
            let blocks = vec![
                OfaBlockConfig {
                    kernel,
                    expand_ratio,
                };
                d
            ];
            units.push(OfaUnit {
                in_ch: cin,
                out_ch: cout,
                stride,
                max_depth,
                active_depth: d,
                blocks,
            });
        }
        OfaSubnet::new(units, self.resolution)
    }
}

// ─── OfaSubnet ───────────────────────────────────────────────────────────────

/// A fully-resolved Once-For-All subnet sliced out of the supernet.
#[derive(Debug, Clone)]
pub struct OfaSubnet {
    /// Per-unit configuration.
    pub units: Vec<OfaUnit>,
    /// Input resolution `(h, w)`.
    pub resolution: (usize, usize),
}

impl OfaSubnet {
    /// Build a subnet, validating that each unit's block list matches its
    /// active depth.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `units` is empty.
    /// - [`NasError::DimensionMismatch`] if a unit's `blocks.len()` differs from
    ///   its `active_depth`, or `active_depth` exceeds `max_depth`.
    pub fn new(units: Vec<OfaUnit>, resolution: (usize, usize)) -> NasResult<Self> {
        if units.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        for u in &units {
            if u.blocks.len() != u.active_depth
                || u.active_depth > u.max_depth
                || u.active_depth == 0
            {
                return Err(NasError::DimensionMismatch {
                    expected: u.active_depth,
                    got: u.blocks.len(),
                });
            }
        }
        Ok(Self { units, resolution })
    }

    /// Total active depth across all units.
    #[must_use]
    pub fn total_depth(&self) -> usize {
        self.units.iter().map(|u| u.active_depth).sum()
    }

    /// Total MAC count of the subnet, threading the resolution through each
    /// unit's stride.
    #[must_use]
    pub fn total_macs(&self) -> u64 {
        let (mut h, mut w) = self.resolution;
        let mut total = 0u64;
        for u in &self.units {
            total = total.saturating_add(u.mac_count(h, w));
            // The unit's first block applies the stride to the resolution.
            let s = u.stride.max(1);
            h /= s;
            w /= s;
        }
        total
    }

    /// Total parameter count of the subnet.
    #[must_use]
    pub fn total_params(&self) -> u64 {
        self.units
            .iter()
            .fold(0u64, |acc, u| acc.saturating_add(u.param_count()))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_space() -> OfaSpace {
        OfaSpace::new(
            vec![
                (16, 24, 2, 4),
                (24, 40, 2, 4),
                (40, 80, 2, 4),
                (80, 96, 1, 4),
            ],
            vec![3, 5, 7],
            vec![3, 4, 6],
            vec![2, 3, 4],
            (224, 224),
        )
        .expect("default OFA space should validate")
    }

    #[test]
    fn schedule_phase_progression() {
        let sched = ShrinkSchedule::new(100).expect("schedule");
        assert_eq!(sched.phase_at(0), ShrinkPhase::FullNetwork);
        assert_eq!(sched.phase_at(50), ShrinkPhase::FullNetwork);
        assert_eq!(sched.phase_at(100), ShrinkPhase::ElasticKernel);
        assert_eq!(sched.phase_at(250), ShrinkPhase::ElasticDepth);
        assert_eq!(sched.phase_at(350), ShrinkPhase::ElasticWidth);
        // Beyond the schedule stays in the last phase.
        assert_eq!(sched.phase_at(10_000), ShrinkPhase::ElasticWidth);
        assert_eq!(sched.total_steps(), 400);
    }

    #[test]
    fn schedule_rejects_zero() {
        assert_eq!(ShrinkSchedule::new(0), Err(NasError::InvalidNumOps));
    }

    #[test]
    fn admissible_collapses_pinned_axes() {
        let space = default_space();
        // Full network: every axis collapses to its maximum.
        let (k, wd, d) = space.admissible(ShrinkPhase::FullNetwork);
        assert_eq!(k, vec![7]);
        assert_eq!(wd, vec![6]);
        assert_eq!(d, vec![4]);
        // Elastic kernel: kernels open, width/depth still pinned.
        let (k, wd, d) = space.admissible(ShrinkPhase::ElasticKernel);
        assert_eq!(k, vec![3, 5, 7]);
        assert_eq!(wd, vec![6]);
        assert_eq!(d, vec![4]);
        // Elastic depth: kernels + depth open, width pinned.
        let (k, wd, d) = space.admissible(ShrinkPhase::ElasticDepth);
        assert_eq!(k, vec![3, 5, 7]);
        assert_eq!(wd, vec![6]);
        assert_eq!(d, vec![2, 3, 4]);
        // Elastic width: everything open.
        let (k, wd, d) = space.admissible(ShrinkPhase::ElasticWidth);
        assert_eq!(k, vec![3, 5, 7]);
        assert_eq!(wd, vec![3, 4, 6]);
        assert_eq!(d, vec![2, 3, 4]);
    }

    #[test]
    fn full_network_sample_is_always_maximal() {
        let space = default_space();
        let mut rng = LcgRng::new(1);
        for _ in 0..20 {
            let net = space
                .sample(ShrinkPhase::FullNetwork, &mut rng)
                .expect("sample");
            for u in &net.units {
                assert_eq!(u.active_depth, 4);
                for b in &u.blocks {
                    assert_eq!(b.kernel, 7);
                    assert_eq!(b.expand_ratio, 6);
                }
            }
        }
    }

    #[test]
    fn elastic_width_sample_respects_choices() {
        let space = default_space();
        let mut rng = LcgRng::new(42);
        for _ in 0..50 {
            let net = space
                .sample(ShrinkPhase::ElasticWidth, &mut rng)
                .expect("sample");
            for u in &net.units {
                assert!(space.depth_choices.contains(&u.active_depth));
                assert_eq!(u.blocks.len(), u.active_depth);
                for b in &u.blocks {
                    assert!(space.kernel_choices.contains(&b.kernel));
                    assert!(space.width_choices.contains(&b.expand_ratio));
                }
            }
        }
    }

    #[test]
    fn max_subnet_dominates_min_subnet() {
        let space = default_space();
        let max = space.max_subnet().expect("max");
        let min = space.min_subnet().expect("min");
        assert!(max.total_depth() > min.total_depth());
        assert!(
            max.total_macs() > min.total_macs(),
            "max {} should exceed min {}",
            max.total_macs(),
            min.total_macs()
        );
        assert!(max.total_params() > min.total_params());
    }

    #[test]
    fn subnet_cost_monotone_in_width() {
        // Two subnets identical except width ratio: larger ratio ⇒ more MACs.
        let units_small = vec![OfaUnit {
            in_ch: 16,
            out_ch: 24,
            stride: 2,
            max_depth: 4,
            active_depth: 2,
            blocks: vec![
                OfaBlockConfig {
                    kernel: 3,
                    expand_ratio: 3,
                };
                2
            ],
        }];
        let units_big = vec![OfaUnit {
            in_ch: 16,
            out_ch: 24,
            stride: 2,
            max_depth: 4,
            active_depth: 2,
            blocks: vec![
                OfaBlockConfig {
                    kernel: 3,
                    expand_ratio: 6,
                };
                2
            ],
        }];
        let small = OfaSubnet::new(units_small, (56, 56)).expect("small");
        let big = OfaSubnet::new(units_big, (56, 56)).expect("big");
        assert!(big.total_macs() > small.total_macs());
        assert!(big.total_params() > small.total_params());
    }

    #[test]
    fn subnet_rejects_block_depth_mismatch() {
        let units = vec![OfaUnit {
            in_ch: 16,
            out_ch: 24,
            stride: 1,
            max_depth: 4,
            active_depth: 3,
            blocks: vec![OfaBlockConfig {
                kernel: 3,
                expand_ratio: 3,
            }], // only 1 block but active_depth=3
        }];
        let r = OfaSubnet::new(units, (32, 32));
        assert!(matches!(r, Err(NasError::DimensionMismatch { .. })));
    }

    #[test]
    fn space_rejects_stage_max_depth_below_choice() {
        // max depth choice is 4 but a stage only reserves 3 slots.
        let r = OfaSpace::new(vec![(16, 24, 1, 3)], vec![3], vec![3], vec![2, 4], (32, 32));
        assert_eq!(r, Err(NasError::InvalidArchEncoding));
    }

    #[test]
    fn sample_is_deterministic_given_seed() {
        let space = default_space();
        let mut a = LcgRng::new(99);
        let mut b = LcgRng::new(99);
        for _ in 0..10 {
            let na = space.sample(ShrinkPhase::ElasticWidth, &mut a).expect("a");
            let nb = space.sample(ShrinkPhase::ElasticWidth, &mut b).expect("b");
            assert_eq!(na.total_depth(), nb.total_depth());
            assert_eq!(na.total_macs(), nb.total_macs());
            assert_eq!(na.total_params(), nb.total_params());
        }
    }
}
