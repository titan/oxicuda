//! Elastic per-rank scaling planner.
//!
//! During serving a cluster may gain or lose a GPU (autoscaling, preemption,
//! failure). This module is the **host-side planner** for that event: given the
//! current TP×SP×EP grid plus the live cache/expert assignment, it recomputes
//! the grid for one-rank-larger or one-rank-smaller topologies and emits a
//! conservation-checked redistribution plan. It is *planning only* — no device
//! synchronisation, no P2P copy — exactly like [`PrefillHandoff`] and the
//! rebalancing layer.
//!
//! [`PrefillHandoff`]: crate::scheduler::disagg_pd::PrefillHandoff
//!
//! ## Scaling model
//!
//! The grid is `world_size = tp · sp · ep`. A *single* rank can be added or
//! removed by adjusting **one elastic axis** by one, which changes the world
//! size by the product of the other two axes. The natural — and tested — elastic
//! configuration holds the two non-elastic axes at `1` (e.g. elastic
//! expert-parallel or elastic data/decode parallel), so incrementing the elastic
//! axis adds *exactly one* rank. The planner is written generally and validates
//! the resulting [`RankCoordinates`] for every rank of the new grid.
//!
//! ## Conservation guarantees
//!
//! * **Experts.** Every one of `n_experts` experts is owned by exactly one rank
//!   before and after. The plan's [`ExpertMove`]s are the (expert → new owner)
//!   diffs; replaying them reproduces the recomputed ownership map.
//! * **Cache.** Every sequence keeps its blocks; sequences are only relocated.
//!   When a rank is removed, *all* of its sequences are redistributed to the
//!   survivors; nothing is dropped. The plan is rejected
//!   ([`DistInferError::RedistributionNotConserved`]) if the total assignment
//!   count would change.

use crate::distributed_cache::partition::{CachePartition, SeqOwnership};
use crate::error::{DistInferError, DistInferResult};
use crate::handle::{ParallelismConfig, RankCoordinates};
use crate::scheduler::rebalance::MigrationMove;

// ─── Axis ─────────────────────────────────────────────────────────────────────

/// The parallelism axis along which elastic scaling occurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElasticAxis {
    /// Scale tensor-parallel degree.
    Tp,
    /// Scale sequence-parallel degree.
    Sp,
    /// Scale expert-parallel degree (the canonical elastic-MoE axis).
    Ep,
}

impl ElasticAxis {
    fn degree(self, cfg: &ParallelismConfig) -> usize {
        match self {
            ElasticAxis::Tp => cfg.tp,
            ElasticAxis::Sp => cfg.sp,
            ElasticAxis::Ep => cfg.ep,
        }
    }

    fn with_degree(self, cfg: &ParallelismConfig, d: usize) -> ParallelismConfig {
        let mut next = *cfg;
        match self {
            ElasticAxis::Tp => next.tp = d,
            ElasticAxis::Sp => next.sp = d,
            ElasticAxis::Ep => next.ep = d,
        }
        next
    }
}

// ─── ExpertMove ───────────────────────────────────────────────────────────────

/// One expert's change of owning rank under a grid resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertMove {
    /// Expert that changes owner.
    pub expert_id: usize,
    /// Rank that owned the expert under the old grid.
    pub from_rank: usize,
    /// Rank that owns the expert under the new grid.
    pub to_rank: usize,
}

// ─── ElasticPlan ──────────────────────────────────────────────────────────────

/// A complete, conservation-checked plan to move from one grid to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElasticPlan {
    /// Grid before the resize.
    pub old_config: ParallelismConfig,
    /// Grid after the resize (`world_size` differs by ±1 in the tested case).
    pub new_config: ParallelismConfig,
    /// Axis that was scaled.
    pub axis: ElasticAxis,
    /// Expert ownership diffs (empty when EP is unchanged / no experts).
    pub expert_moves: Vec<ExpertMove>,
    /// Cache sequence relocations that re-level (or evacuate) the partition.
    pub cache_moves: Vec<MigrationMove>,
}

impl ElasticPlan {
    /// World size of the new grid.
    #[must_use]
    pub fn new_world_size(&self) -> usize {
        self.new_config.world_size()
    }

    /// Whether the resize added a rank (`true`) or removed one (`false`).
    #[must_use]
    pub fn is_scale_up(&self) -> bool {
        self.new_config.world_size() > self.old_config.world_size()
    }

    /// Number of experts that change owner.
    #[must_use]
    pub fn n_expert_moves(&self) -> usize {
        self.expert_moves.len()
    }

    /// Number of sequences relocated in the cache.
    #[must_use]
    pub fn n_cache_moves(&self) -> usize {
        self.cache_moves.len()
    }

    /// Total KV blocks relocated by the cache portion of the plan.
    #[must_use]
    pub fn cache_blocks_moved(&self) -> usize {
        self.cache_moves.iter().map(|m| m.n_blocks).sum()
    }
}

// ─── ElasticScaler ────────────────────────────────────────────────────────────

/// Host-side planner that resizes the parallelism grid by one rank.
///
/// All methods are pure: they read the current grid + assignment and return a
/// plan. Mutating the live [`CachePartition`] is the caller's explicit follow-up
/// via [`ElasticScaler::apply_cache_moves`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticScaler {
    /// Total number of MoE experts spread across the EP axis (`0` = no MoE).
    n_experts: usize,
}

impl ElasticScaler {
    /// Construct a scaler for a model with `n_experts` MoE experts (`0` if the
    /// model is dense and only cache redistribution is planned).
    #[must_use]
    pub fn new(n_experts: usize) -> Self {
        Self { n_experts }
    }

    /// Number of experts the scaler accounts for.
    #[must_use]
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Compute, for a given grid, the owning rank of each expert.
    ///
    /// Experts are partitioned contiguously across the EP axis: rank `e`
    /// (ep-rank) owns experts `[e·k, (e+1)·k)` where `k = n_experts / ep`. The
    /// *flat global rank* of an expert's owner is the rank whose `ep_rank`
    /// equals that block index and whose other coords are `0` (the EP group
    /// leader), matching [`RankCoordinates::peer_ep`].
    ///
    /// # Errors
    ///
    /// [`DistInferError::EpExpertsMisaligned`] if `n_experts` is not divisible
    /// by `ep`.
    pub fn expert_owners(&self, cfg: &ParallelismConfig) -> DistInferResult<Vec<usize>> {
        if self.n_experts == 0 {
            return Ok(Vec::new());
        }
        let ep = cfg.ep;
        if !ep.divides_nonzero(self.n_experts) {
            return Err(DistInferError::EpExpertsMisaligned {
                n_experts: self.n_experts,
                degree: ep,
            });
        }
        let per_rank = self.n_experts / ep;
        let mut owners = Vec::with_capacity(self.n_experts);
        for e in 0..self.n_experts {
            let ep_rank = e / per_rank;
            // EP-group leader: tp_rank = sp_rank = 0, ep_rank = block.
            let coords = RankCoordinates {
                tp_rank: 0,
                sp_rank: 0,
                ep_rank,
                global_rank: 0,
            };
            owners.push(coords.to_global(cfg));
        }
        Ok(owners)
    }

    /// Plan the addition of a rank along `axis`.
    ///
    /// Returns the new grid plus the expert + cache redistribution required to
    /// occupy it. With the two non-`axis` degrees at `1`, the new world size is
    /// exactly `old + 1`.
    ///
    /// # Errors
    ///
    /// * [`DistInferError::EpExpertsMisaligned`] if scaling EP would make
    ///   `n_experts` indivisible by the new degree.
    /// * Propagates partition/coordinate validation errors.
    pub fn plan_add_rank(
        &self,
        cfg: &ParallelismConfig,
        axis: ElasticAxis,
        partition: &CachePartition,
    ) -> DistInferResult<ElasticPlan> {
        let new_degree = axis.degree(cfg) + 1;
        self.plan_resize(cfg, axis, new_degree, partition)
    }

    /// Plan the removal of a rank along `axis`.
    ///
    /// The highest-indexed rank on `axis` is evacuated; its experts and
    /// sequences are redistributed to the survivors.
    ///
    /// # Errors
    ///
    /// * [`DistInferError::CannotScaleBelowOne`] if the resulting grid would
    ///   have a zero degree.
    /// * [`DistInferError::EpExpertsMisaligned`] if EP would become indivisible.
    pub fn plan_remove_rank(
        &self,
        cfg: &ParallelismConfig,
        axis: ElasticAxis,
        partition: &CachePartition,
    ) -> DistInferResult<ElasticPlan> {
        let cur = axis.degree(cfg);
        if cur <= 1 {
            return Err(DistInferError::CannotScaleBelowOne {
                world_size: cfg.world_size(),
            });
        }
        self.plan_resize(cfg, axis, cur - 1, partition)
    }

    /// Core resize: validate the new grid, diff expert ownership, and re-level
    /// (or evacuate) the cache, then verify conservation.
    fn plan_resize(
        &self,
        cfg: &ParallelismConfig,
        axis: ElasticAxis,
        new_degree: usize,
        partition: &CachePartition,
    ) -> DistInferResult<ElasticPlan> {
        if new_degree == 0 {
            return Err(DistInferError::CannotScaleBelowOne {
                world_size: cfg.world_size(),
            });
        }
        let new_cfg = axis.with_degree(cfg, new_degree);
        new_cfg.validate()?;
        // Validate that every rank of the new grid has well-formed coordinates.
        let new_ws = new_cfg.world_size();
        for r in 0..new_ws {
            RankCoordinates::from_global(r, &new_cfg)?;
        }

        // ── Experts ───────────────────────────────────────────────────────────
        let expert_moves = if matches!(axis, ElasticAxis::Ep) && self.n_experts > 0 {
            let old_owners = self.expert_owners(cfg)?;
            let new_owners = self.expert_owners(&new_cfg)?;
            let mut moves = Vec::new();
            for (e, (&from, &to)) in old_owners.iter().zip(new_owners.iter()).enumerate() {
                if from != to {
                    moves.push(ExpertMove {
                        expert_id: e,
                        from_rank: from,
                        to_rank: to,
                    });
                }
            }
            moves
        } else {
            Vec::new()
        };

        // ── Cache ─────────────────────────────────────────────────────────────
        let cache_moves = self.plan_cache_redistribution(partition, cfg, &new_cfg)?;

        Ok(ElasticPlan {
            old_config: *cfg,
            new_config: new_cfg,
            axis,
            expert_moves,
            cache_moves,
        })
    }

    /// Re-level the sequence ownership across the new rank set.
    ///
    /// Sequences on ranks that no longer exist (shrink) are forcibly evacuated;
    /// otherwise sequences move from over-target ranks to under-target ranks
    /// until each rank holds within one sequence of the fair share. The result
    /// is a list of [`MigrationMove`]s whose total count equals the number of
    /// sequences that changed owner — and which conserves the live block total.
    fn plan_cache_redistribution(
        &self,
        partition: &CachePartition,
        old_cfg: &ParallelismConfig,
        new_cfg: &ParallelismConfig,
    ) -> DistInferResult<Vec<MigrationMove>> {
        let new_ws = new_cfg.world_size();
        let old_ws = old_cfg.world_size();
        let mut owners: Vec<SeqOwnership> = partition.ownerships();
        owners.sort_by_key(|o| o.seq_id); // deterministic ordering

        let n_seqs = owners.len();
        if n_seqs == 0 || new_ws == 0 {
            return Ok(Vec::new());
        }

        // Per-rank sequence count over the *new* rank index space.
        let mut counts = vec![0usize; new_ws.max(old_ws)];
        for o in &owners {
            counts[o.owner_rank] += 1;
        }

        // Target fair share over the new ranks.
        let base = n_seqs / new_ws;
        let rem = n_seqs % new_ws;
        let target = |rank: usize| -> usize { if rank < rem { base + 1 } else { base } };

        let mut moves = Vec::new();

        // 1. Evacuate any rank that no longer exists (rank >= new_ws on shrink).
        //    Greedily place each orphaned sequence onto the currently-emptiest
        //    surviving rank (relative to its target).
        let orphan_indices: Vec<usize> = owners
            .iter()
            .enumerate()
            .filter(|(_, o)| o.owner_rank >= new_ws)
            .map(|(i, _)| i)
            .collect();
        for idx in orphan_indices {
            let from = owners[idx].owner_rank;
            // Coldest surviving rank relative to its target (room to spare).
            let to = (0..new_ws)
                .min_by_key(|&r| counts[r] as isize - target(r) as isize)
                .unwrap_or(0);
            counts[from] -= 1;
            counts[to] += 1;
            moves.push(MigrationMove {
                seq_id: owners[idx].seq_id,
                from_rank: from,
                to_rank: to,
                n_blocks: owners[idx].n_blocks,
            });
            owners[idx].owner_rank = to;
        }

        // 2. Level the surviving ranks: move from over-target to under-target.
        loop {
            // Hottest over-target rank (within new range).
            let donor = (0..new_ws)
                .filter(|&r| counts[r] > target(r))
                .max_by_key(|&r| counts[r] - target(r));
            let Some(donor) = donor else { break };
            // Coldest under-target rank.
            let recv = (0..new_ws)
                .filter(|&r| counts[r] < target(r))
                .min_by_key(|&r| counts[r]);
            let Some(recv) = recv else { break };
            if donor == recv {
                break;
            }
            // Relocate one (smallest-block) sequence from donor to recv.
            let victim = owners
                .iter()
                .enumerate()
                .filter(|(_, o)| o.owner_rank == donor)
                .min_by_key(|(_, o)| o.n_blocks)
                .map(|(i, _)| i);
            let Some(victim) = victim else { break };
            counts[donor] -= 1;
            counts[recv] += 1;
            moves.push(MigrationMove {
                seq_id: owners[victim].seq_id,
                from_rank: donor,
                to_rank: recv,
                n_blocks: owners[victim].n_blocks,
            });
            owners[victim].owner_rank = recv;
        }

        // ── Conservation check: same set of sequences, none dropped/duplicated.
        let final_total: usize = counts[..new_ws].iter().sum();
        if final_total != n_seqs {
            return Err(DistInferError::RedistributionNotConserved {
                expected: n_seqs,
                got: final_total,
            });
        }
        // No sequence may remain stranded on a defunct rank.
        if owners.iter().any(|o| o.owner_rank >= new_ws) {
            return Err(DistInferError::RedistributionNotConserved {
                expected: n_seqs,
                got: final_total,
            });
        }

        Ok(moves)
    }

    /// Apply only the cache portion of a plan to a live partition.
    ///
    /// For scale-up the partition must already have been re-sized to the new
    /// world (more ranks); for scale-down callers typically re-level *before*
    /// dropping the physical rank. Returns the number of relocations performed.
    ///
    /// # Errors
    ///
    /// Propagates [`CachePartition::apply_migration`] errors.
    pub fn apply_cache_moves(
        plan: &ElasticPlan,
        partition: &mut CachePartition,
    ) -> DistInferResult<usize> {
        for mv in &plan.cache_moves {
            partition.apply_migration(mv.seq_id, mv.from_rank, mv.to_rank)?;
        }
        Ok(plan.cache_moves.len())
    }
}

// ─── Small divisibility helper ─────────────────────────────────────────────────

/// `usize` extension: does `self` evenly divide `n` (with `self` non-zero)?
trait DivisorExt {
    fn divides_nonzero(self, n: usize) -> bool;
}

impl DivisorExt for usize {
    #[inline]
    fn divides_nonzero(self, n: usize) -> bool {
        self != 0 && n % self == 0
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{DistInferHandle, ParallelismConfig, SmVersion};

    fn handle_world(ws: usize, ep: usize) -> DistInferHandle {
        // Put the whole world on the EP axis for elastic-MoE tests.
        let tp = ws / ep;
        DistInferHandle::new(0, SmVersion(80), 0, ParallelismConfig { tp, sp: 1, ep })
            .expect("handle should construct")
    }

    fn ep_cfg(ep: usize) -> ParallelismConfig {
        ParallelismConfig { tp: 1, sp: 1, ep }
    }

    fn partition_with_seqs(ws: usize, n_seqs: u64, blocks: usize) -> CachePartition {
        let h = handle_world(ws, ws); // ep == ws so all ranks are EP-distinct
        let mut part = CachePartition::new(h, &vec![1000usize; ws], 0.2).expect("new");
        for s in 0..n_seqs {
            part.assign(s, blocks).expect("assign");
        }
        part
    }

    #[test]
    fn expert_owners_partition_contiguously() {
        let scaler = ElasticScaler::new(8);
        // ep=4, tp=sp=1 → 2 experts/rank, owner global == ep_rank.
        let owners = scaler.expert_owners(&ep_cfg(4)).expect("owners");
        assert_eq!(owners, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn expert_owners_misaligned_errors() {
        let scaler = ElasticScaler::new(7); // 7 not divisible by ep=4
        assert!(matches!(
            scaler.expert_owners(&ep_cfg(4)),
            Err(DistInferError::EpExpertsMisaligned { .. })
        ));
    }

    #[test]
    fn add_rank_yields_valid_larger_grid() {
        let scaler = ElasticScaler::new(0);
        let part = partition_with_seqs(4, 0, 10); // empty cache, just grid math
        let plan = scaler
            .plan_add_rank(&ep_cfg(4), ElasticAxis::Ep, &part)
            .expect("add");
        assert_eq!(plan.new_config, ep_cfg(5));
        assert_eq!(plan.new_world_size(), 5, "N+1 grid");
        assert!(plan.is_scale_up());
        // Every rank of the new grid has valid coordinates (checked inside; here
        // we assert the world size grew by exactly one).
        assert_eq!(plan.new_world_size(), part.stats().len() + 1);
    }

    #[test]
    fn add_rank_conserves_and_levels_experts() {
        // 12 experts on ep=3 (4/rank) → ep=4 (3/rank). Every expert reassigned to
        // exactly one owner; multiset of owners is exactly {0,1,2,3} × counts.
        let scaler = ElasticScaler::new(12);
        let part = partition_with_seqs(3, 0, 10);
        let plan = scaler
            .plan_add_rank(&ep_cfg(3), ElasticAxis::Ep, &part)
            .expect("add");
        assert_eq!(plan.new_config, ep_cfg(4));

        // Replay expert moves over the old ownership and confirm we land exactly
        // on the freshly-computed new ownership (conservation of every expert).
        let mut owners = scaler.expert_owners(&ep_cfg(3)).expect("old");
        for mv in &plan.expert_moves {
            assert_eq!(
                owners[mv.expert_id], mv.from_rank,
                "move starts where expert is"
            );
            owners[mv.expert_id] = mv.to_rank;
        }
        let expected = scaler.expert_owners(&ep_cfg(4)).expect("new");
        assert_eq!(
            owners, expected,
            "replayed moves reproduce the new ownership"
        );

        // Each of the 4 new ranks owns exactly 3 experts.
        for rank in 0..4 {
            let owned = owners.iter().filter(|&&o| o == rank).count();
            assert_eq!(owned, 3, "rank {rank} must own 12/4 = 3 experts");
        }
    }

    #[test]
    fn remove_rank_redistributes_experts_without_loss() {
        // 12 experts on ep=4 → ep=3. Rank 3's experts must be re-homed; total
        // expert count preserved.
        let scaler = ElasticScaler::new(12);
        let part = partition_with_seqs(4, 0, 10);
        let plan = scaler
            .plan_remove_rank(&ep_cfg(4), ElasticAxis::Ep, &part)
            .expect("remove");
        assert_eq!(plan.new_config, ep_cfg(3));
        assert!(!plan.is_scale_up());

        let mut owners = scaler.expert_owners(&ep_cfg(4)).expect("old");
        for mv in &plan.expert_moves {
            owners[mv.expert_id] = mv.to_rank;
        }
        let expected = scaler.expert_owners(&ep_cfg(3)).expect("new");
        assert_eq!(owners, expected);
        // No expert left on the now-defunct rank 3.
        assert!(owners.iter().all(|&o| o < 3), "rank 3 fully evacuated");
        // Conservation: still 12 experts, each owned exactly once.
        assert_eq!(owners.len(), 12);
        for rank in 0..3 {
            assert_eq!(owners.iter().filter(|&&o| o == rank).count(), 4);
        }
    }

    #[test]
    fn add_rank_cache_levels_onto_new_rank() {
        // 4 ranks, 8 sequences (2 each after assign). Grow to 5 ranks: the new
        // rank (4) must receive some sequences; total conserved.
        let scaler = ElasticScaler::new(0);
        let part = partition_with_seqs(4, 8, 10);
        let total_seqs = part.ownerships().len();
        let plan = scaler
            .plan_add_rank(&ep_cfg(4), ElasticAxis::Ep, &part)
            .expect("add");

        // The new rank index 4 should appear as a destination.
        assert!(
            plan.cache_moves.iter().any(|m| m.to_rank == 4),
            "scale-up must populate the new rank"
        );
        // Conservation: the planner verified internally; assert externally too by
        // replaying onto rank counts.
        let mut counts = [0usize; 5];
        for o in part.ownerships() {
            counts[o.owner_rank] += 1;
        }
        for mv in &plan.cache_moves {
            counts[mv.from_rank] -= 1;
            counts[mv.to_rank] += 1;
        }
        assert_eq!(counts.iter().sum::<usize>(), total_seqs, "no sequence lost");
        // Post-plan distribution is leveled to within one across 5 ranks.
        let max = *counts.iter().max().expect("nonempty");
        let min = *counts.iter().min().expect("nonempty");
        assert!(
            max - min <= 1,
            "leveled within one seq, got spread {max}-{min}"
        );
    }

    #[test]
    fn remove_rank_evacuates_all_its_sequences() {
        // 4 ranks, sequences spread; remove rank 3 → none may remain on rank 3.
        let scaler = ElasticScaler::new(0);
        let part = partition_with_seqs(4, 8, 10);
        let total_seqs = part.ownerships().len();
        let plan = scaler
            .plan_remove_rank(&ep_cfg(4), ElasticAxis::Ep, &part)
            .expect("remove");
        assert_eq!(plan.new_world_size(), 3);

        // Replay and confirm rank 3 is empty and total is conserved.
        let mut owner_of: std::collections::HashMap<u64, usize> = part
            .ownerships()
            .into_iter()
            .map(|o| (o.seq_id, o.owner_rank))
            .collect();
        for mv in &plan.cache_moves {
            assert_eq!(owner_of[&mv.seq_id], mv.from_rank);
            owner_of.insert(mv.seq_id, mv.to_rank);
        }
        assert_eq!(owner_of.len(), total_seqs, "no sequence dropped");
        assert!(
            owner_of.values().all(|&r| r < 3),
            "every sequence lives on a surviving rank"
        );
    }

    #[test]
    fn cannot_remove_below_single_rank() {
        let scaler = ElasticScaler::new(0);
        let part = partition_with_seqs(1, 1, 10);
        assert!(matches!(
            scaler.plan_remove_rank(&ep_cfg(1), ElasticAxis::Ep, &part),
            Err(DistInferError::CannotScaleBelowOne { .. })
        ));
    }

    #[test]
    fn add_rank_ep_misalignment_errors() {
        // 6 experts on ep=3 (ok) → ep=4 makes 6 indivisible → error surfaced.
        let scaler = ElasticScaler::new(6);
        let part = partition_with_seqs(3, 0, 10);
        assert!(matches!(
            scaler.plan_add_rank(&ep_cfg(3), ElasticAxis::Ep, &part),
            Err(DistInferError::EpExpertsMisaligned { .. })
        ));
    }

    #[test]
    fn apply_cache_moves_executes_on_live_partition_scale_down() {
        // Build a 4-rank partition, plan a removal of rank 3, then actually apply
        // the cache moves (capacity is ample so apply_migration succeeds) and
        // confirm rank 3 ends empty with all blocks conserved.
        let scaler = ElasticScaler::new(0);
        let mut part = partition_with_seqs(4, 8, 10);
        let blocks_before: usize = part.stats().iter().map(|s| s.used_blocks()).sum();
        let plan = scaler
            .plan_remove_rank(&ep_cfg(4), ElasticAxis::Ep, &part)
            .expect("remove");
        let applied = ElasticScaler::apply_cache_moves(&plan, &mut part).expect("apply");
        assert_eq!(applied, plan.n_cache_moves());
        assert_eq!(part.stats()[3].n_seqs, 0, "rank 3 evacuated");
        let blocks_after: usize = part.stats().iter().map(|s| s.used_blocks()).sum();
        assert_eq!(blocks_before, blocks_after, "blocks conserved on apply");
    }

    #[test]
    fn scale_up_on_tp_axis_also_valid() {
        // Elastic along TP with sp=ep=1: tp=2 → tp=3 is a +1 grid.
        let scaler = ElasticScaler::new(0);
        let cfg = ParallelismConfig {
            tp: 2,
            sp: 1,
            ep: 1,
        };
        let h = DistInferHandle::new(0, SmVersion(80), 0, cfg).expect("handle");
        let mut part = CachePartition::new(h, &[1000, 1000], 0.2).expect("new");
        for s in 0..4u64 {
            part.assign(s, 10).expect("assign");
        }
        let plan = scaler
            .plan_add_rank(&cfg, ElasticAxis::Tp, &part)
            .expect("add");
        assert_eq!(
            plan.new_config,
            ParallelismConfig {
                tp: 3,
                sp: 1,
                ep: 1
            }
        );
        assert_eq!(plan.new_world_size(), 3);
        // No experts (dense) → no expert moves, but cache levels onto rank 2.
        assert!(plan.expert_moves.is_empty());
        assert!(plan.cache_moves.iter().any(|m| m.to_rank == 2));
    }
}
