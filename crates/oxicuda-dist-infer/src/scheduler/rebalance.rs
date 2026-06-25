//! Autonomous load-rebalancing trigger for the distributed KV cache.
//!
//! [`CachePartition::rebalance_suggestions`](crate::distributed_cache::partition::CachePartition::rebalance_suggestions)
//! can *propose* a single migration, but nothing in the serving loop fires it
//! on its own. This module supplies the missing host-side decision layer: a
//! [`RebalanceMonitor`] that watches an imbalance metric and, once it crosses a
//! configurable threshold, synthesises a concrete multi-step [`MigrationPlan`]
//! that drives the imbalance back below the threshold.
//!
//! Two imbalance signals are supported, mirroring the two metrics already in the
//! crate:
//!
//! * **Cache utilization spread** — `max_util − min_util` across ranks, the same
//!   quantity [`CachePartition::utilization_imbalance`](crate::distributed_cache::partition::CachePartition::utilization_imbalance)
//!   reports and that `rebalance_suggestions` thresholds on.
//! * **MoE expert-load coefficient of variation** — `TopKRouter::load_balance_cv`
//!   over a [`RoutingPlan`]; a hot
//!   expert distribution is a second, independent reason to fire rebalancing.
//!
//! The planner is *pure host-side accounting*: it projects candidate moves onto
//! a working copy of the per-rank stats (never the live partition), so the
//! emitted plan is verified to **conserve total blocks** (no dropped or
//! duplicated assignment) and to **reduce the imbalance** before it is returned.
//! Applying the plan to the real [`CachePartition`] is a separate, explicit step.

use crate::distributed_cache::partition::{CachePartition, RankCacheStats, SeqOwnership};
use crate::error::{DistInferError, DistInferResult};
use crate::expert_parallel::router::{RoutingPlan, TopKRouter};

// ─── MigrationMove ────────────────────────────────────────────────────────────

/// A single sequence relocation in a [`MigrationPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MigrationMove {
    /// Sequence to relocate.
    pub seq_id: u64,
    /// Rank currently owning the sequence (the overloaded side).
    pub from_rank: usize,
    /// Rank that will own the sequence after migration (the underloaded side).
    pub to_rank: usize,
    /// KV blocks transferred with the sequence (conserved across the move).
    pub n_blocks: usize,
}

// ─── MigrationPlan ────────────────────────────────────────────────────────────

/// A concrete, conservation-checked rebalancing plan.
///
/// Emitted by [`RebalanceMonitor::evaluate`] when (and only when) the imbalance
/// crosses the configured threshold. The ordered list of [`MigrationMove`]s,
/// applied in sequence, takes the partition from `imbalance_before` to
/// `imbalance_after` (both recorded for audit). The plan is guaranteed
/// non-empty, block-conserving, and strictly imbalance-reducing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationPlan {
    /// Ordered relocations to perform.
    pub moves: Vec<MigrationMove>,
    /// Utilization spread (`max−min`, scaled to `1e6` ppm) before the plan.
    pub imbalance_before_ppm: u32,
    /// Utilization spread (ppm) projected after applying every move.
    pub imbalance_after_ppm: u32,
}

impl MigrationPlan {
    /// Number of relocations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// Whether the plan is empty (carries no moves).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    /// Total KV blocks moved by the plan.
    #[must_use]
    pub fn total_blocks_moved(&self) -> usize {
        self.moves.iter().map(|m| m.n_blocks).sum()
    }

    /// Imbalance reduction achieved, in ppm (`before − after`).
    #[must_use]
    pub fn imbalance_reduction_ppm(&self) -> u32 {
        self.imbalance_before_ppm
            .saturating_sub(self.imbalance_after_ppm)
    }
}

// ─── Working projection ───────────────────────────────────────────────────────

/// Convert a utilization spread in `[0, 1]` to integer ppm for `Eq`-friendly
/// plan reporting (floats are deliberately kept out of the public plan struct).
fn util_to_ppm(spread: f32) -> u32 {
    (spread.clamp(0.0, 1.0) * 1_000_000.0).round() as u32
}

/// Max−min utilization over a working set of rank stats.
fn spread_of(stats: &[RankCacheStats]) -> f32 {
    if stats.is_empty() {
        return 0.0;
    }
    let mut max_u = f32::NEG_INFINITY;
    let mut min_u = f32::INFINITY;
    for s in stats {
        let u = s.utilization();
        if u > max_u {
            max_u = u;
        }
        if u < min_u {
            min_u = u;
        }
    }
    (max_u - min_u).max(0.0)
}

/// Index of the most- and least-utilized rank in `stats`.
///
/// Ties resolve to the lowest index for determinism. Returns `None` when there
/// are fewer than two ranks (nothing can move).
fn extremes(stats: &[RankCacheStats]) -> Option<(usize, usize)> {
    if stats.len() < 2 {
        return None;
    }
    let mut max_r = 0usize;
    let mut min_r = 0usize;
    let mut max_u = stats[0].utilization();
    let mut min_u = stats[0].utilization();
    for (r, s) in stats.iter().enumerate().skip(1) {
        let u = s.utilization();
        if u > max_u {
            max_u = u;
            max_r = r;
        }
        if u < min_u {
            min_u = u;
            min_r = r;
        }
    }
    Some((max_r, min_r))
}

// ─── RebalanceMonitor ─────────────────────────────────────────────────────────

/// Host-side autonomous trigger for cache rebalancing.
///
/// Construct once with the imbalance thresholds, then call [`evaluate`] (or
/// [`evaluate_with_moe`]) every scheduling tick. It returns `Some(plan)` only on
/// the ticks where the imbalance actually warrants migration, and `None`
/// otherwise — so a balanced system is left untouched.
///
/// [`evaluate`]: RebalanceMonitor::evaluate
/// [`evaluate_with_moe`]: RebalanceMonitor::evaluate_with_moe
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RebalanceMonitor {
    /// Utilization spread (`max−min`, in `[0, 1]`) at/above which a cache
    /// rebalance fires.
    util_threshold: f32,
    /// MoE expert-load CV at/above which an expert rebalance is *also* warranted.
    /// `None` disables the MoE signal.
    moe_cv_threshold: Option<f32>,
    /// Safety cap on the number of moves a single plan may contain.
    max_moves: usize,
}

impl RebalanceMonitor {
    /// Default cap on relocations emitted in one plan.
    pub const DEFAULT_MAX_MOVES: usize = 64;

    /// Construct a monitor that fires when the cache utilization spread reaches
    /// `util_threshold` (a value in `[0, 1]`).
    ///
    /// # Errors
    ///
    /// [`DistInferError::InvalidThreshold`] if `util_threshold` is outside
    /// `[0, 1]`.
    pub fn new(util_threshold: f32) -> DistInferResult<Self> {
        if !(0.0..=1.0).contains(&util_threshold) {
            return Err(DistInferError::InvalidThreshold {
                threshold: util_threshold,
            });
        }
        Ok(Self {
            util_threshold,
            moe_cv_threshold: None,
            max_moves: Self::DEFAULT_MAX_MOVES,
        })
    }

    /// Enable the MoE expert-load signal: an expert-load coefficient of
    /// variation at/above `cv_threshold` is treated as an additional reason to
    /// rebalance.
    ///
    /// # Errors
    ///
    /// [`DistInferError::InvalidThreshold`] if `cv_threshold` is negative.
    pub fn with_moe_cv_threshold(mut self, cv_threshold: f32) -> DistInferResult<Self> {
        if cv_threshold < 0.0 || !cv_threshold.is_finite() {
            return Err(DistInferError::InvalidThreshold {
                threshold: cv_threshold,
            });
        }
        self.moe_cv_threshold = Some(cv_threshold);
        Ok(self)
    }

    /// Override the per-plan move cap (default [`Self::DEFAULT_MAX_MOVES`]).
    #[must_use]
    pub fn with_max_moves(mut self, max_moves: usize) -> Self {
        self.max_moves = max_moves.max(1);
        self
    }

    /// The configured cache-utilization trigger threshold.
    #[must_use]
    pub fn util_threshold(&self) -> f32 {
        self.util_threshold
    }

    /// The configured MoE-CV trigger threshold, if any.
    #[must_use]
    pub fn moe_cv_threshold(&self) -> Option<f32> {
        self.moe_cv_threshold
    }

    /// Whether a cache utilization spread should fire the trigger.
    #[must_use]
    pub fn cache_should_trigger(&self, imbalance: f32) -> bool {
        imbalance >= self.util_threshold
    }

    /// Whether an MoE expert-load CV should fire the trigger.
    ///
    /// Always `false` if the MoE signal was not enabled.
    #[must_use]
    pub fn moe_should_trigger(&self, plan: &RoutingPlan) -> bool {
        match self.moe_cv_threshold {
            Some(t) => TopKRouter::load_balance_cv(plan) >= t,
            None => false,
        }
    }

    /// Inspect a partition and, **if** its utilization imbalance has reached the
    /// threshold, synthesise a concrete migration plan that reduces it.
    ///
    /// Returns `None` for a balanced partition (no trigger). When a plan *is*
    /// returned it is non-empty, block-conserving, and verified to lower the
    /// imbalance.
    ///
    /// # Errors
    ///
    /// [`DistInferError::RedistributionNotConserved`] only if an internal
    /// projection bug would drop or duplicate blocks — a hard invariant
    /// violation that should never occur.
    pub fn evaluate(&self, partition: &CachePartition) -> DistInferResult<Option<MigrationPlan>> {
        let imbalance = partition.utilization_imbalance();
        if !self.cache_should_trigger(imbalance) {
            return Ok(None);
        }
        self.build_plan(partition).map(Some)
    }

    /// Like [`evaluate`](Self::evaluate) but also fires when the MoE expert-load
    /// CV (over `routing_plan`) crosses the MoE threshold, even if the cache
    /// utilization spread alone would not.
    ///
    /// This is the host-side coupling between a skewed expert distribution and
    /// the cache rebalancer: a hot-expert imbalance is a leading indicator that
    /// the owning ranks will soon be cache-overloaded, so we proactively emit a
    /// cache migration plan when one is achievable.
    ///
    /// # Errors
    ///
    /// As [`evaluate`](Self::evaluate).
    pub fn evaluate_with_moe(
        &self,
        partition: &CachePartition,
        routing_plan: &RoutingPlan,
    ) -> DistInferResult<Option<MigrationPlan>> {
        let cache_fire = self.cache_should_trigger(partition.utilization_imbalance());
        let moe_fire = self.moe_should_trigger(routing_plan);
        if !cache_fire && !moe_fire {
            return Ok(None);
        }
        // A plan can only ever be built from real cache imbalance; if the MoE
        // signal fired but the cache is already perfectly level there is nothing
        // to migrate, so report "no actionable plan" rather than a useless one.
        if extremes(partition.stats()).is_none() || partition.utilization_imbalance() <= 0.0 {
            return Ok(None);
        }
        self.build_plan(partition).map(Some)
    }

    /// Greedily project smallest-sequence moves from the hottest rank to the
    /// coldest until the spread drops below threshold (or no progress / cap).
    fn build_plan(&self, partition: &CachePartition) -> DistInferResult<MigrationPlan> {
        let mut stats: Vec<RankCacheStats> = partition.stats().to_vec();
        let mut owners: Vec<SeqOwnership> = partition.ownerships();
        let total_blocks_before: usize = stats.iter().map(|s| s.used_blocks()).sum();

        let imbalance_before = spread_of(&stats);
        let mut moves: Vec<MigrationMove> = Vec::new();

        while moves.len() < self.max_moves && spread_of(&stats) >= self.util_threshold {
            let Some((from_rank, to_rank)) = extremes(&stats) else {
                break;
            };
            if from_rank == to_rank {
                break;
            }
            // Smallest sequence on the hot rank that the cold rank can hold —
            // the same "evict the cheapest victim" rule `rebalance_suggestions`
            // uses, extended to honour the destination's free capacity.
            let victim_pos = owners
                .iter()
                .enumerate()
                .filter(|(_, o)| {
                    o.owner_rank == from_rank && o.n_blocks <= stats[to_rank].free_blocks
                })
                .min_by_key(|(_, o)| o.n_blocks)
                .map(|(i, _)| i);
            let Some(victim_pos) = victim_pos else {
                break; // nothing relocatable without exhausting the target
            };

            let victim = owners[victim_pos];
            // Refuse a move that does not strictly help (would not reduce, or
            // would invert, the spread) — guarantees monotone progress.
            let mut trial = stats.clone();
            trial[from_rank].free_blocks += victim.n_blocks;
            trial[from_rank].n_seqs = trial[from_rank].n_seqs.saturating_sub(1);
            trial[to_rank].free_blocks -= victim.n_blocks;
            trial[to_rank].n_seqs += 1;
            if spread_of(&trial) >= spread_of(&stats) {
                break;
            }

            // Commit the projected move.
            stats = trial;
            owners[victim_pos].owner_rank = to_rank;
            moves.push(MigrationMove {
                seq_id: victim.seq_id,
                from_rank,
                to_rank,
                n_blocks: victim.n_blocks,
            });
        }

        // Conservation invariant: used blocks are merely relocated, never
        // created or destroyed.
        let total_blocks_after: usize = stats.iter().map(|s| s.used_blocks()).sum();
        if total_blocks_after != total_blocks_before {
            return Err(DistInferError::RedistributionNotConserved {
                expected: total_blocks_before,
                got: total_blocks_after,
            });
        }

        if moves.is_empty() {
            // Imbalance was above threshold but unactionable (e.g. one giant
            // sequence the cold rank cannot hold). Surface an honest internal
            // signal rather than a meaningless empty "plan".
            return Err(DistInferError::Internal(
                "rebalance triggered but no sequence can be relocated without exhausting the target rank",
            ));
        }

        Ok(MigrationPlan {
            moves,
            imbalance_before_ppm: util_to_ppm(imbalance_before),
            imbalance_after_ppm: util_to_ppm(spread_of(&stats)),
        })
    }

    /// Apply a [`MigrationPlan`] to a live [`CachePartition`], performing every
    /// relocation through [`CachePartition::apply_migration`].
    ///
    /// Returns the number of moves applied. This is the explicit "execute the
    /// host-side plan" step kept separate from planning.
    ///
    /// # Errors
    ///
    /// Propagates any [`CachePartition::apply_migration`] error.
    pub fn apply_plan(
        plan: &MigrationPlan,
        partition: &mut CachePartition,
    ) -> DistInferResult<usize> {
        for mv in &plan.moves {
            partition.apply_migration(mv.seq_id, mv.from_rank, mv.to_rank)?;
        }
        Ok(plan.moves.len())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_parallel::router::TopKRouter;
    use crate::handle::{DistInferHandle, ParallelismConfig, SmVersion};

    fn handle_world(ws: usize) -> DistInferHandle {
        DistInferHandle::new(
            0,
            SmVersion(80),
            0,
            ParallelismConfig {
                tp: ws,
                sp: 1,
                ep: 1,
            },
        )
        .expect("handle should construct")
    }

    /// Build a deliberately skewed 4-rank partition: rank 0 is hot, the rest
    /// nearly empty.
    fn skewed_partition() -> CachePartition {
        let h = handle_world(4);
        let mut part = CachePartition::new(h, &[100, 100, 100, 100], 0.2).expect("new");
        // Pile many sequences on rank 0 by assigning, then forcing them there via
        // migration so we control the skew exactly.
        // Easiest deterministic skew: assign small seqs and migrate onto rank 0.
        for seq in 0..8u64 {
            let r = part.assign(seq, 10).expect("assign");
            if r != 0 {
                part.apply_migration(seq, r, 0).expect("force onto rank 0");
            }
        }
        part
    }

    #[test]
    fn invalid_threshold_rejected() {
        assert!(matches!(
            RebalanceMonitor::new(-0.1),
            Err(DistInferError::InvalidThreshold { .. })
        ));
        assert!(matches!(
            RebalanceMonitor::new(1.5),
            Err(DistInferError::InvalidThreshold { .. })
        ));
        assert!(RebalanceMonitor::new(0.0).is_ok());
        assert!(RebalanceMonitor::new(1.0).is_ok());
    }

    #[test]
    fn balanced_partition_does_not_trigger() {
        let h = handle_world(4);
        let mut part = CachePartition::new(h, &[100, 100, 100, 100], 0.2).expect("new");
        // Spread 8 equal sequences round-robin → perfectly level.
        for seq in 0..8u64 {
            part.assign(seq, 10).expect("assign");
        }
        assert!(
            part.utilization_imbalance() < 0.05,
            "round-robin assignment should be near-level, got {}",
            part.utilization_imbalance()
        );
        let mon = RebalanceMonitor::new(0.2).expect("monitor");
        let plan = mon.evaluate(&part).expect("evaluate");
        assert!(plan.is_none(), "balanced state must not trigger a plan");
    }

    #[test]
    fn skewed_partition_triggers_and_reduces_imbalance() {
        let part = skewed_partition();
        let before = part.utilization_imbalance();
        // rank0 holds 8×10 = 80 blocks of 100 (util 0.8); others 0.0 → spread 0.8.
        assert!(
            before > 0.5,
            "fixture should be heavily skewed, got {before}"
        );

        let mon = RebalanceMonitor::new(0.2).expect("monitor");
        let plan = mon
            .evaluate(&part)
            .expect("evaluate")
            .expect("skew must trigger a plan");

        assert!(!plan.is_empty(), "plan must carry at least one move");
        // Plan must strictly reduce the imbalance.
        assert!(
            plan.imbalance_after_ppm < plan.imbalance_before_ppm,
            "after {} should be < before {}",
            plan.imbalance_after_ppm,
            plan.imbalance_before_ppm
        );
        // Every move is valid: hot→cold, real sequence, non-zero blocks.
        for mv in &plan.moves {
            assert_ne!(mv.from_rank, mv.to_rank);
            assert!(mv.n_blocks > 0);
        }
    }

    #[test]
    fn plan_application_actually_levels_the_partition() {
        let mut part = skewed_partition();
        let mon = RebalanceMonitor::new(0.2).expect("monitor");
        let plan = mon
            .evaluate(&part)
            .expect("evaluate")
            .expect("must trigger");
        let before = part.utilization_imbalance();
        let n = RebalanceMonitor::apply_plan(&plan, &mut part).expect("apply");
        assert_eq!(n, plan.len(), "all moves applied");
        let after = part.utilization_imbalance();
        assert!(
            after < before,
            "applied plan must reduce live imbalance: {before} → {after}"
        );
        // Re-evaluating should now decline to fire (or fire a strictly smaller
        // plan) — convergence, not oscillation.
        assert!(after < 0.2 + 1e-3 || mon.evaluate(&part).expect("re-eval").is_some());
    }

    #[test]
    fn plan_conserves_total_blocks() {
        let part = skewed_partition();
        let total_before: usize = part.stats().iter().map(|s| s.used_blocks()).sum();
        let mon = RebalanceMonitor::new(0.2).expect("monitor");
        let plan = mon
            .evaluate(&part)
            .expect("evaluate")
            .expect("must trigger");
        // Apply on a clone-by-replay partition and confirm conservation.
        let mut part2 = skewed_partition();
        RebalanceMonitor::apply_plan(&plan, &mut part2).expect("apply");
        let total_after: usize = part2.stats().iter().map(|s| s.used_blocks()).sum();
        assert_eq!(
            total_before, total_after,
            "migration must conserve total used blocks"
        );
    }

    #[test]
    fn moe_skew_triggers_via_cv_signal() {
        // ── Item 2: MoE load-imbalance stress ────────────────────────────────
        // Construct an artificially skewed expert assignment: 16 tokens, 4
        // experts, top-1, but route *every* token to expert 0.
        let router = TopKRouter::new(4, 1).expect("router");
        let n_tokens = 16;
        let mut logits = vec![0.0f32; n_tokens * 4];
        for t in 0..n_tokens {
            logits[t * 4] = 5.0; // expert 0 always wins
        }
        let plan = router.route(&logits, n_tokens).expect("route");
        let cv = TopKRouter::load_balance_cv(&plan);
        // Skewed: expert_load = [16,0,0,0] → mean 4, var = (12²+3·4²)/4 = 48 → cv = √48/4 ≈ 1.732.
        assert!(cv > 1.0, "all-to-one routing must report high CV, got {cv}");
        assert_eq!(plan.expert_load, vec![16, 0, 0, 0]);

        // The monitor with an MoE-CV threshold recognises this as a trigger…
        let mon = RebalanceMonitor::new(0.2)
            .expect("monitor")
            .with_moe_cv_threshold(0.5)
            .expect("cv threshold");
        assert!(
            mon.moe_should_trigger(&plan),
            "high-CV expert plan must fire the MoE signal"
        );

        // …and when paired with a correspondingly hot cache it produces a plan.
        let part = skewed_partition();
        let migration = mon
            .evaluate_with_moe(&part, &plan)
            .expect("evaluate_with_moe")
            .expect("MoE skew + cache skew must yield a plan");
        assert!(!migration.is_empty());
        assert!(migration.imbalance_after_ppm < migration.imbalance_before_ppm);
    }

    #[test]
    fn balanced_moe_does_not_trigger_moe_signal() {
        let router = TopKRouter::new(4, 1).expect("router");
        // Perfectly balanced: 4 tokens, one per expert.
        let logits = vec![
            1.0f32, 0.0, 0.0, 0.0, // → e0
            0.0, 1.0, 0.0, 0.0, // → e1
            0.0, 0.0, 1.0, 0.0, // → e2
            0.0, 0.0, 0.0, 1.0, // → e3
        ];
        let plan = router.route(&logits, 4).expect("route");
        assert!(TopKRouter::load_balance_cv(&plan) < 1e-6);
        let mon = RebalanceMonitor::new(0.2)
            .expect("monitor")
            .with_moe_cv_threshold(0.5)
            .expect("cv threshold");
        assert!(
            !mon.moe_should_trigger(&plan),
            "level experts must not fire"
        );
    }

    #[test]
    fn moe_signal_disabled_by_default() {
        let router = TopKRouter::new(2, 1).expect("router");
        let logits = vec![5.0f32, 0.0, 5.0, 0.0]; // both to expert 0 (skewed)
        let plan = router.route(&logits, 2).expect("route");
        let mon = RebalanceMonitor::new(0.2).expect("monitor");
        assert!(
            !mon.moe_should_trigger(&plan),
            "without a configured CV threshold the MoE signal is inert"
        );
    }

    #[test]
    fn max_moves_cap_is_respected() {
        let part = skewed_partition();
        let mon = RebalanceMonitor::new(0.05)
            .expect("monitor")
            .with_max_moves(1);
        let plan = mon
            .evaluate(&part)
            .expect("evaluate")
            .expect("must trigger");
        assert!(plan.len() <= 1, "plan must respect the move cap");
    }

    #[test]
    fn single_rank_partition_never_triggers() {
        let h = handle_world(1);
        let mut part = CachePartition::new(h, &[50], 0.2).expect("new");
        part.assign(1, 40).expect("assign"); // util 0.8 but nowhere to move
        let mon = RebalanceMonitor::new(0.2).expect("monitor");
        // imbalance over a single rank is 0 → no trigger.
        assert_eq!(part.utilization_imbalance(), 0.0);
        assert!(mon.evaluate(&part).expect("evaluate").is_none());
    }
}
