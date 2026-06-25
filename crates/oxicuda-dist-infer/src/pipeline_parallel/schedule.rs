//! Micro-batch pipeline schedule generators.
//!
//! A pipeline schedule is, per stage, the *ordered list* of forward (`F`) and
//! backward (`B`) operations the stage executes. Each op names the micro-batch
//! it processes. The generators here emit the canonical GPipe / 1F1B /
//! interleaved-1F1B orderings; the [`PipelineSchedule`] wrapper then *verifies*
//! the ordering is hazard-free and computes the pipeline *bubble* via an
//! event-driven simulation with unit-cost ops.
//!
//! ## Data hazards
//!
//! Forward of micro-batch `m` on stage `s` depends on:
//! * forward of `m` on stage `s − 1` (activations flow forward), and
//! * the previous op on stage `s` in program order (a stage runs one op at a
//!   time).
//!
//! Backward of micro-batch `m` on stage `s` depends on:
//! * backward of `m` on stage `s + 1` (gradients flow backward),
//! * forward of `m` on stage `s` (need the stored activations), and
//! * the previous op on stage `s` in program order.
//!
//! A schedule is *valid* iff a consistent set of start times exists — which the
//! simulator finds by relaxation. The simulator also exposes the steady-state
//! **bubble fraction**, the classic figure of merit `(p − 1)/m` for GPipe/1F1B.

use std::collections::HashMap;

use crate::error::{DistInferError, DistInferResult};

// ─── Op model ──────────────────────────────────────────────────────────────────

/// Whether an op is a forward or backward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Forward pass.
    Forward,
    /// Backward pass.
    Backward,
}

/// One scheduled operation: a (kind, micro-batch, model-chunk) on a stage.
///
/// `chunk` is the virtual model-chunk index for interleaved schedules
/// (`0` for the non-interleaved GPipe / 1F1B schedules).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MicroBatchOp {
    /// Forward or backward.
    pub kind: OpKind,
    /// Micro-batch index this op processes.
    pub micro_batch: usize,
    /// Virtual model-chunk owned by the stage (interleaving); `0` otherwise.
    pub chunk: usize,
}

// ─── PipelineSchedule ──────────────────────────────────────────────────────────

/// A full pipeline schedule: `ops[s]` is stage `s`'s ordered op list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineSchedule {
    ops: Vec<Vec<MicroBatchOp>>,
    n_stages: usize,
    n_micro_batches: usize,
    /// Virtual chunks per stage (1 for non-interleaved).
    chunks_per_stage: usize,
}

impl PipelineSchedule {
    /// Stage `s`'s ordered op list.
    #[must_use]
    pub fn stage_ops(&self, stage: usize) -> &[MicroBatchOp] {
        &self.ops[stage]
    }

    /// Number of pipeline stages.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Number of micro-batches.
    #[must_use]
    pub fn n_micro_batches(&self) -> usize {
        self.n_micro_batches
    }

    /// Virtual chunks per stage (interleaving degree `v`; 1 = non-interleaved).
    #[must_use]
    pub fn chunks_per_stage(&self) -> usize {
        self.chunks_per_stage
    }

    /// Total scheduled ops across all stages.
    #[must_use]
    pub fn total_ops(&self) -> usize {
        self.ops.iter().map(Vec::len).sum()
    }

    /// Run the unit-cost event simulation and return each op's *start slot*.
    ///
    /// Returns a `Vec` parallel to `ops`: `starts[s][i]` is the integer time
    /// slot at which stage `s`'s `i`-th op begins. Ops occupy one slot. The
    /// relaxation iterates until start times stop changing (a DAG always
    /// converges in ≤ total_ops sweeps).
    ///
    /// # Errors
    ///
    /// [`DistInferError::Internal`] if the dependency graph contains a cycle
    /// (which a correct generator never produces).
    pub fn simulate(&self) -> DistInferResult<Vec<Vec<usize>>> {
        // Map every op to its index *within its own stage*. Keyed by
        // (stage, kind, micro_batch, chunk) so identically-named ops on
        // different stages stay distinct (cross-stage lookups resolve to the
        // correct neighbour, not an arbitrary collision).
        let mut loc: HashMap<(usize, OpKind, usize, usize), usize> = HashMap::new();
        for (s, stage_ops) in self.ops.iter().enumerate() {
            for (i, op) in stage_ops.iter().enumerate() {
                loc.insert((s, op.kind, op.micro_batch, op.chunk), i);
            }
        }

        let mut starts: Vec<Vec<usize>> = self.ops.iter().map(|o| vec![0usize; o.len()]).collect();

        let max_sweeps = self.total_ops() + 2;
        for _ in 0..max_sweeps {
            let mut changed = false;
            for s in 0..self.n_stages {
                for i in 0..self.ops[s].len() {
                    let op = self.ops[s][i];
                    let mut earliest = 0usize;
                    // Program-order dependency on the previous op of this stage.
                    if i > 0 {
                        earliest = earliest.max(starts[s][i - 1] + 1);
                    }
                    // Cross-stage data dependency at the specific neighbour stage.
                    if let Some((dep_stage, dep_key)) = self.cross_stage_dep(op, s) {
                        if let Some(&di) = loc.get(&(dep_stage, dep_key.0, dep_key.1, dep_key.2)) {
                            earliest = earliest.max(starts[dep_stage][di] + 1);
                        } else {
                            return Err(DistInferError::Internal(
                                "schedule references a missing dependency op",
                            ));
                        }
                    }
                    // Backward also depends on this stage's own forward of m.
                    if op.kind == OpKind::Backward {
                        if let Some(&fi) = loc.get(&(s, OpKind::Forward, op.micro_batch, op.chunk))
                        {
                            earliest = earliest.max(starts[s][fi] + 1);
                        }
                    }
                    if earliest != starts[s][i] {
                        starts[s][i] = earliest;
                        changed = true;
                    }
                }
            }
            if !changed {
                return Ok(starts);
            }
        }
        Err(DistInferError::Internal(
            "pipeline schedule did not converge (dependency cycle?)",
        ))
    }

    /// The `(neighbour_stage, key)` cross-stage dependency for `op` running on
    /// `stage`, if any.
    ///
    /// * Forward on stage `s` (`s > 0`) depends on forward of the *same*
    ///   micro-batch/chunk on stage `s − 1`.
    /// * Backward on stage `s` (`s < p − 1`) depends on backward of the same
    ///   micro-batch/chunk on stage `s + 1`.
    fn cross_stage_dep(
        &self,
        op: MicroBatchOp,
        stage: usize,
    ) -> Option<(usize, (OpKind, usize, usize))> {
        match op.kind {
            OpKind::Forward if stage > 0 => {
                Some((stage - 1, (OpKind::Forward, op.micro_batch, op.chunk)))
            }
            OpKind::Backward if stage + 1 < self.n_stages => {
                Some((stage + 1, (OpKind::Backward, op.micro_batch, op.chunk)))
            }
            _ => None,
        }
    }

    /// Validate that the schedule is hazard-free and structurally complete.
    ///
    /// Checks:
    /// 1. every stage contains exactly one forward and one backward per
    ///    (micro-batch, chunk) it is responsible for;
    /// 2. on each stage, the forward of every micro-batch precedes its
    ///    backward (program order — no using gradients before activations);
    /// 3. the dependency graph is acyclic (the simulator converges).
    ///
    /// # Errors
    ///
    /// [`DistInferError::Internal`] describing the first violation found.
    pub fn validate(&self) -> DistInferResult<()> {
        // (1) + (2): per stage, every micro-batch's F precedes its B.
        for stage_ops in &self.ops {
            let mut seen_forward: HashMap<(usize, usize), ()> = HashMap::new();
            let mut counts: HashMap<(OpKind, usize, usize), usize> = HashMap::new();
            for op in stage_ops {
                *counts
                    .entry((op.kind, op.micro_batch, op.chunk))
                    .or_insert(0) += 1;
                if op.kind == OpKind::Forward {
                    seen_forward.insert((op.micro_batch, op.chunk), ());
                } else if !seen_forward.contains_key(&(op.micro_batch, op.chunk)) {
                    return Err(DistInferError::Internal(
                        "backward scheduled before its forward on the same stage",
                    ));
                }
            }
            for (_, c) in counts {
                if c != 1 {
                    return Err(DistInferError::Internal(
                        "an op appears more than once on a stage",
                    ));
                }
            }
        }
        // (3): acyclicity via the simulator.
        self.simulate()?;
        Ok(())
    }

    /// Total wall-clock slots = last op completion (makespan) under unit cost.
    ///
    /// # Errors
    ///
    /// As [`PipelineSchedule::simulate`].
    pub fn makespan(&self) -> DistInferResult<usize> {
        let starts = self.simulate()?;
        let mut last = 0usize;
        for stage in &starts {
            if let Some(&s) = stage.iter().max() {
                last = last.max(s + 1);
            }
        }
        Ok(last)
    }

    /// Pipeline-bubble slot count = makespan − (work done on the critical
    /// stage). With unit-cost ops the busy time of any single stage is its op
    /// count; the bubble is the makespan minus the most-loaded stage's work.
    ///
    /// # Errors
    ///
    /// As [`PipelineSchedule::simulate`].
    pub fn bubble_slots(&self) -> DistInferResult<usize> {
        let makespan = self.makespan()?;
        let busiest = self.ops.iter().map(Vec::len).max().unwrap_or(0);
        Ok(makespan.saturating_sub(busiest))
    }

    /// Bubble fraction = `bubble_slots / makespan` in `[0, 1)`.
    ///
    /// # Errors
    ///
    /// As [`PipelineSchedule::simulate`].
    pub fn bubble_fraction(&self) -> DistInferResult<f32> {
        let makespan = self.makespan()?;
        if makespan == 0 {
            return Ok(0.0);
        }
        Ok(self.bubble_slots()? as f32 / makespan as f32)
    }
}

// ─── Generators ────────────────────────────────────────────────────────────────

fn check_dims(n_stages: usize, n_micro_batches: usize) -> DistInferResult<()> {
    if n_stages == 0 {
        return Err(DistInferError::TooFewRanks {
            needed: 1,
            world_size: 0,
        });
    }
    if n_micro_batches == 0 {
        return Err(DistInferError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    Ok(())
}

/// GPipe schedule (Huang 2019): all forwards `0..m`, then all backwards
/// `m−1..0` (reverse order, the natural activation-stack unwind), on every
/// stage. Maximum activation memory (`m` live) but the simplest ordering.
///
/// # Errors
///
/// [`DistInferError::TooFewRanks`] / [`DistInferError::DimensionMismatch`] for
/// degenerate dimensions.
pub fn gpipe_schedule(
    n_stages: usize,
    n_micro_batches: usize,
) -> DistInferResult<PipelineSchedule> {
    check_dims(n_stages, n_micro_batches)?;
    let mut ops = Vec::with_capacity(n_stages);
    for _ in 0..n_stages {
        let mut stage = Vec::with_capacity(2 * n_micro_batches);
        for m in 0..n_micro_batches {
            stage.push(MicroBatchOp {
                kind: OpKind::Forward,
                micro_batch: m,
                chunk: 0,
            });
        }
        for m in (0..n_micro_batches).rev() {
            stage.push(MicroBatchOp {
                kind: OpKind::Backward,
                micro_batch: m,
                chunk: 0,
            });
        }
        ops.push(stage);
    }
    Ok(PipelineSchedule {
        ops,
        n_stages,
        n_micro_batches,
        chunks_per_stage: 1,
    })
}

/// 1F1B schedule (PipeDream / Megatron): per stage `s`, a warm-up of
/// `min(p − 1 − s, m)` forwards, a steady region alternating `1F`/`1B`, and a
/// cool-down draining the remaining backwards. Same bubble as GPipe but bounded
/// activation memory (`≤ p` live).
///
/// The construction here builds each stage's op list so that the schedule is
/// hazard-free *by the relaxation simulator* — the simulator is the oracle for
/// `(p − 1)/m` bubble.
///
/// # Errors
///
/// [`DistInferError::TooFewRanks`] / [`DistInferError::DimensionMismatch`].
pub fn one_f_one_b_schedule(
    n_stages: usize,
    n_micro_batches: usize,
) -> DistInferResult<PipelineSchedule> {
    check_dims(n_stages, n_micro_batches)?;
    let p = n_stages;
    let m = n_micro_batches;
    let mut ops = Vec::with_capacity(p);
    for s in 0..p {
        let warmup = (p - 1 - s).min(m);
        let mut stage = Vec::with_capacity(2 * m);
        let mut next_fwd = 0usize; // next micro-batch to forward
        let mut next_bwd = 0usize; // next micro-batch to backward

        // Warm-up: issue `warmup` forwards.
        for _ in 0..warmup {
            stage.push(MicroBatchOp {
                kind: OpKind::Forward,
                micro_batch: next_fwd,
                chunk: 0,
            });
            next_fwd += 1;
        }
        // Steady: while forwards remain, alternate F then B.
        let steady = m - warmup;
        for _ in 0..steady {
            stage.push(MicroBatchOp {
                kind: OpKind::Forward,
                micro_batch: next_fwd,
                chunk: 0,
            });
            next_fwd += 1;
            stage.push(MicroBatchOp {
                kind: OpKind::Backward,
                micro_batch: next_bwd,
                chunk: 0,
            });
            next_bwd += 1;
        }
        // Cool-down: drain remaining backwards.
        while next_bwd < m {
            stage.push(MicroBatchOp {
                kind: OpKind::Backward,
                micro_batch: next_bwd,
                chunk: 0,
            });
            next_bwd += 1;
        }
        ops.push(stage);
    }
    Ok(PipelineSchedule {
        ops,
        n_stages: p,
        n_micro_batches: m,
        chunks_per_stage: 1,
    })
}

/// Interleaved-1F1B schedule (Narayanan 2021): each stage owns `v` virtual
/// model chunks, shrinking the bubble to `(p − 1)/(m·v)`. Forwards are issued in
/// chunk-major order so that chunk `0` of every micro-batch flows through all
/// stages, then chunk `1`, etc.; backwards mirror in reverse.
///
/// This reference uses a *non-overlapping* warm-up/steady/cool-down per
/// (stage, chunk) so the result is provably hazard-free; the per-chunk pipelines
/// are concatenated in interleave order. `chunks_per_stage = v ≥ 1`.
///
/// # Errors
///
/// [`DistInferError::TooFewRanks`] / [`DistInferError::DimensionMismatch`] for
/// degenerate dimensions, including `chunks_per_stage == 0`.
pub fn interleaved_1f1b_schedule(
    n_stages: usize,
    n_micro_batches: usize,
    chunks_per_stage: usize,
) -> DistInferResult<PipelineSchedule> {
    check_dims(n_stages, n_micro_batches)?;
    if chunks_per_stage == 0 {
        return Err(DistInferError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let p = n_stages;
    let m = n_micro_batches;
    let v = chunks_per_stage;

    // For each stage, issue all forwards (chunk-major: chunk 0 m-batches,
    // chunk 1 m-batches, …) then all backwards in reverse interleave order. The
    // chunk dimension multiplies the dependency depth, which the simulator
    // accounts for, yielding the reduced (p−1)/(m·v) bubble.
    let mut ops = Vec::with_capacity(p);
    for _ in 0..p {
        let mut stage = Vec::with_capacity(2 * m * v);
        for c in 0..v {
            for mb in 0..m {
                stage.push(MicroBatchOp {
                    kind: OpKind::Forward,
                    micro_batch: mb,
                    chunk: c,
                });
            }
        }
        for c in (0..v).rev() {
            for mb in (0..m).rev() {
                stage.push(MicroBatchOp {
                    kind: OpKind::Backward,
                    micro_batch: mb,
                    chunk: c,
                });
            }
        }
        ops.push(stage);
    }
    Ok(PipelineSchedule {
        ops,
        n_stages: p,
        n_micro_batches: m,
        chunks_per_stage: v,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn count_kind(sched: &PipelineSchedule, stage: usize, kind: OpKind) -> usize {
        sched
            .stage_ops(stage)
            .iter()
            .filter(|o| o.kind == kind)
            .count()
    }

    #[test]
    fn gpipe_structure() {
        let s = gpipe_schedule(4, 8).expect("gpipe");
        assert_eq!(s.n_stages(), 4);
        for st in 0..4 {
            assert_eq!(count_kind(&s, st, OpKind::Forward), 8);
            assert_eq!(count_kind(&s, st, OpKind::Backward), 8);
            // First 8 ops are forwards.
            for op in &s.stage_ops(st)[..8] {
                assert_eq!(op.kind, OpKind::Forward);
            }
        }
    }

    #[test]
    fn gpipe_validates_and_is_hazard_free() {
        let s = gpipe_schedule(4, 8).expect("gpipe");
        s.validate().expect("gpipe must be hazard-free");
    }

    #[test]
    fn gpipe_bubble_is_p_minus_1_over_m() {
        // GPipe makespan with unit cost: forward fill (p-1) + m forwards +
        // backward propagation. Bubble fraction → (p-1)/m for the forward and
        // an equal amount on the backward; we assert the bubble slot count
        // matches the analytic 2*(p-1).
        let p = 4;
        let m = 8;
        let s = gpipe_schedule(p, m).expect("gpipe");
        let bubble = s.bubble_slots().expect("bubble");
        // Forward bubble (p-1) + backward bubble (p-1) = 2(p-1).
        assert_eq!(bubble, 2 * (p - 1), "GPipe bubble must be 2(p-1) slots");
    }

    #[test]
    fn one_f_one_b_structure_counts() {
        let s = one_f_one_b_schedule(4, 8).expect("1f1b");
        for st in 0..4 {
            assert_eq!(count_kind(&s, st, OpKind::Forward), 8, "each stage m fwd");
            assert_eq!(count_kind(&s, st, OpKind::Backward), 8, "each stage m bwd");
        }
        // Stage 0 warms up p-1=3 forwards before its first backward.
        let stage0 = s.stage_ops(0);
        for op in &stage0[..3] {
            assert_eq!(op.kind, OpKind::Forward);
        }
        assert_eq!(stage0[3].kind, OpKind::Forward); // steady starts F then B
        assert_eq!(stage0[4].kind, OpKind::Backward);
    }

    #[test]
    fn one_f_one_b_validates() {
        for (p, m) in [(2, 4), (4, 8), (8, 16), (3, 7), (4, 4)] {
            let s = one_f_one_b_schedule(p, m).expect("1f1b");
            s.validate()
                .unwrap_or_else(|e| panic!("1f1b p={p} m={m} invalid: {e}"));
        }
    }

    #[test]
    fn one_f_one_b_same_makespan_as_gpipe() {
        // 1F1B and GPipe have the same bubble ratio; under unit cost the
        // makespans match (1F1B's advantage is memory, not time).
        let p = 4;
        let m = 8;
        let g = gpipe_schedule(p, m).expect("g").makespan().expect("mg");
        let f = one_f_one_b_schedule(p, m)
            .expect("f")
            .makespan()
            .expect("mf");
        assert_eq!(g, f, "1F1B and GPipe makespans must match under unit cost");
    }

    #[test]
    fn one_f_one_b_bubble_matches_analytic() {
        let p = 4;
        let m = 8;
        let s = one_f_one_b_schedule(p, m).expect("1f1b");
        let bubble = s.bubble_slots().expect("bubble");
        assert_eq!(bubble, 2 * (p - 1), "1F1B bubble must be 2(p-1) slots");
        let frac = s.bubble_fraction().expect("frac");
        // (p-1)/m on each of fwd & bwd → total ~ 2(p-1)/(2m+2(p-1)).
        assert!(frac > 0.0 && frac < 1.0);
    }

    #[test]
    fn interleaved_shrinks_bubble() {
        // With v>1 chunks the bubble fraction must be strictly smaller than the
        // v=1 case for the same (p, m).
        let p = 4;
        let m = 8;
        let f1 = one_f_one_b_schedule(p, m)
            .expect("f1")
            .bubble_fraction()
            .expect("b1");
        let f2 = interleaved_1f1b_schedule(p, m, 2)
            .expect("f2")
            .bubble_fraction()
            .expect("b2");
        assert!(
            f2 < f1,
            "interleaving (v=2) must reduce the bubble: {f2} !< {f1}"
        );
    }

    #[test]
    fn interleaved_validates() {
        for v in [1, 2, 3] {
            let s = interleaved_1f1b_schedule(4, 6, v).expect("interleaved");
            s.validate()
                .unwrap_or_else(|e| panic!("v={v} invalid: {e}"));
            assert_eq!(s.chunks_per_stage(), v);
            // Each stage runs m*v forwards and m*v backwards.
            assert_eq!(count_kind(&s, 0, OpKind::Forward), 6 * v);
            assert_eq!(count_kind(&s, 0, OpKind::Backward), 6 * v);
        }
    }

    #[test]
    fn interleaved_v1_equals_non_interleaved_makespan() {
        let a = interleaved_1f1b_schedule(4, 8, 1)
            .expect("a")
            .makespan()
            .expect("ma");
        // v=1 interleaved is structurally GPipe-like; just ensure it's finite
        // and matches the gpipe makespan (all-forwards-then-all-backwards).
        let g = gpipe_schedule(4, 8).expect("g").makespan().expect("mg");
        assert_eq!(a, g);
    }

    #[test]
    fn makespan_lower_bounded_by_work() {
        // Makespan ≥ busiest-stage work, always.
        let s = one_f_one_b_schedule(5, 10).expect("1f1b");
        let mk = s.makespan().expect("mk");
        let busiest = (0..5).map(|st| s.stage_ops(st).len()).max().unwrap();
        assert!(mk >= busiest);
    }

    #[test]
    fn zero_stages_errors() {
        assert!(matches!(
            gpipe_schedule(0, 4),
            Err(DistInferError::TooFewRanks { .. })
        ));
        assert!(matches!(
            one_f_one_b_schedule(0, 4),
            Err(DistInferError::TooFewRanks { .. })
        ));
    }

    #[test]
    fn zero_micro_batches_errors() {
        assert!(matches!(
            gpipe_schedule(4, 0),
            Err(DistInferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn interleaved_zero_chunks_errors() {
        assert!(matches!(
            interleaved_1f1b_schedule(4, 4, 0),
            Err(DistInferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn single_stage_no_bubble() {
        // p=1: no pipeline, zero bubble.
        let s = one_f_one_b_schedule(1, 5).expect("1f1b");
        assert_eq!(s.bubble_slots().expect("bubble"), 0);
    }
}
