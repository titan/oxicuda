//! Expert-parallel all-to-all dispatch / combine for distributed MoE.
//!
//! In expert-parallel MoE the `E` experts are sharded across `D` device groups
//! (`E` must be a multiple of `D`; each device hosts `E/D` experts). A token
//! routed to expert `e` must be physically sent to the device that owns `e`.
//! This is realised on real hardware by two **all-to-all** collectives:
//!
//! * **dispatch** (combine-scatter): every device sends each of its local
//!   tokens to the destination device that owns the token's expert, packing
//!   them into per-(src,dst) capacity-bounded buffers;
//! * **combine** (scatter-gather): after the experts run, the processed tokens
//!   are sent back along the reverse path and scattered into their original
//!   token slots, weighted by the gate score.
//!
//! This module provides a faithful single-process *simulation* of that data
//! movement (no real NCCL): it builds the all-to-all send buffers, enforces a
//! per-device expert capacity (dropping overflow exactly like the GPU path),
//! and exposes [`ExpertParallelPlan`] describing the routing so a caller can
//! drive the local expert compute and then combine.

use crate::error::{MoeError, MoeResult};

/// Configuration for expert-parallel dispatch.
#[derive(Debug, Clone)]
pub struct ExpertParallelConfig {
    /// Total number of experts `E` (must be divisible by `n_devices`).
    pub n_experts: usize,
    /// Number of device groups `D`.
    pub n_devices: usize,
    /// Capacity factor for the per-device token buffer.
    pub capacity_factor: f32,
    /// Minimum per-device capacity (floor).
    pub min_capacity: usize,
}

/// Where a single token ended up after dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenPlacement {
    /// Destination device that owns the token's expert (`usize::MAX` if dropped).
    pub device: usize,
    /// Local expert index *within* the destination device (`0..E/D`).
    pub local_expert: usize,
    /// Slot index inside the destination device's buffer.
    pub slot: usize,
}

impl TokenPlacement {
    /// A dropped token (overflowed its destination device's capacity).
    pub const DROPPED: Self = Self {
        device: usize::MAX,
        local_expert: 0,
        slot: 0,
    };

    /// Whether this token was dropped.
    #[must_use]
    pub fn is_dropped(&self) -> bool {
        self.device == usize::MAX
    }
}

/// Plan describing the all-to-all routing of every token.
#[derive(Debug, Clone)]
pub struct ExpertParallelPlan {
    /// Per-token placement, length `n_tokens`.
    pub placements: Vec<TokenPlacement>,
    /// Number of experts hosted per device (`E / D`).
    pub experts_per_device: usize,
    /// Per-device token capacity (a device buffer holds
    /// `experts_per_device * capacity` tokens).
    pub capacity: usize,
    /// Number of device groups.
    pub n_devices: usize,
    /// Total dropped tokens.
    pub n_dropped: usize,
    /// Tokens received per device (`[n_devices]`), excluding drops.
    pub tokens_per_device: Vec<usize>,
}

impl ExpertParallelPlan {
    /// Flat slot offset of a placement inside its device buffer:
    /// `local_expert * capacity + slot`.
    #[must_use]
    pub fn device_buffer_offset(&self, p: &TokenPlacement) -> usize {
        p.local_expert * self.capacity + p.slot
    }
}

/// Build the dispatch plan from per-token (global) expert assignments.
///
/// # Errors
/// Returns [`MoeError::InvalidExpertCount`] if `n_experts == 0` or `n_devices == 0`,
/// [`MoeError::DispatchFailed`] if `n_experts` is not divisible by `n_devices`,
/// [`MoeError::InvalidCapacityFactor`] for a non-positive factor,
/// [`MoeError::EmptyInput`] for zero tokens,
/// [`MoeError::DimensionMismatch`] on assignment length, and
/// [`MoeError::ExpertIndexOutOfRange`] for an out-of-range expert.
pub fn build_dispatch_plan(
    expert_assignments: &[usize],
    n_tokens: usize,
    cfg: &ExpertParallelConfig,
) -> MoeResult<ExpertParallelPlan> {
    if cfg.n_experts == 0 || cfg.n_devices == 0 {
        return Err(MoeError::InvalidExpertCount {
            n_experts: cfg.n_experts,
        });
    }
    if !cfg.n_experts.is_multiple_of(cfg.n_devices) {
        return Err(MoeError::DispatchFailed {
            msg: format!(
                "n_experts {} not divisible by n_devices {}",
                cfg.n_experts, cfg.n_devices
            ),
        });
    }
    if !cfg.capacity_factor.is_finite() || cfg.capacity_factor <= 0.0 {
        return Err(MoeError::InvalidCapacityFactor {
            factor: cfg.capacity_factor,
        });
    }
    if n_tokens == 0 {
        return Err(MoeError::EmptyInput);
    }
    if expert_assignments.len() != n_tokens {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: expert_assignments.len(),
        });
    }

    let experts_per_device = cfg.n_experts / cfg.n_devices;
    // Per-expert capacity, mirroring Switch: ceil(T / E * cap_factor).
    let raw_cap = (n_tokens as f32 / cfg.n_experts as f32 * cfg.capacity_factor).ceil() as usize;
    let capacity = raw_cap.max(cfg.min_capacity).max(1);

    // Slot counters per (device, local_expert).
    let mut slot_counts = vec![0usize; cfg.n_devices * experts_per_device];
    let mut placements = vec![TokenPlacement::DROPPED; n_tokens];
    let mut n_dropped = 0usize;
    let mut tokens_per_device = vec![0usize; cfg.n_devices];

    for (tok, &expert) in expert_assignments.iter().enumerate() {
        if expert == usize::MAX {
            n_dropped += 1;
            continue;
        }
        if expert >= cfg.n_experts {
            return Err(MoeError::ExpertIndexOutOfRange {
                idx: expert,
                n_experts: cfg.n_experts,
            });
        }
        let device = expert / experts_per_device;
        let local_expert = expert % experts_per_device;
        let counter = &mut slot_counts[device * experts_per_device + local_expert];
        if *counter < capacity {
            placements[tok] = TokenPlacement {
                device,
                local_expert,
                slot: *counter,
            };
            *counter += 1;
            tokens_per_device[device] += 1;
        } else {
            n_dropped += 1;
        }
    }

    Ok(ExpertParallelPlan {
        placements,
        experts_per_device,
        capacity,
        n_devices: cfg.n_devices,
        n_dropped,
        tokens_per_device,
    })
}

/// All-to-all **dispatch**: pack token features into per-device buffers.
///
/// Returns `n_devices` buffers, each of shape
/// `[experts_per_device * capacity * d_model]`, zero-padded for empty slots.
///
/// # Errors
/// Returns [`MoeError::DimensionMismatch`] if `x` is not `[n_tokens × d_model]`.
pub fn dispatch_all_to_all(
    x: &[f32],
    plan: &ExpertParallelPlan,
    n_tokens: usize,
    d_model: usize,
) -> MoeResult<Vec<Vec<f32>>> {
    if d_model == 0 {
        return Err(MoeError::InvalidHiddenDim { dim: d_model });
    }
    if x.len() != n_tokens * d_model {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens * d_model,
            got: x.len(),
        });
    }
    let buf_rows = plan.experts_per_device * plan.capacity;
    let mut buffers = vec![vec![0.0_f32; buf_rows * d_model]; plan.n_devices];
    for (tok, p) in plan.placements.iter().enumerate() {
        if p.is_dropped() {
            continue;
        }
        let offset = plan.device_buffer_offset(p) * d_model;
        buffers[p.device][offset..offset + d_model]
            .copy_from_slice(&x[tok * d_model..(tok + 1) * d_model]);
    }
    Ok(buffers)
}

/// All-to-all **combine**: scatter processed per-device buffers back to token
/// space, scaling each token by `scores[token]` if provided.
///
/// # Errors
/// Returns [`MoeError::DimensionMismatch`] on buffer-count / shape / score
/// mismatches.
pub fn combine_all_to_all(
    device_buffers: &[Vec<f32>],
    plan: &ExpertParallelPlan,
    n_tokens: usize,
    d_model: usize,
    scores: Option<&[f32]>,
) -> MoeResult<Vec<f32>> {
    if d_model == 0 {
        return Err(MoeError::InvalidHiddenDim { dim: d_model });
    }
    if device_buffers.len() != plan.n_devices {
        return Err(MoeError::DimensionMismatch {
            expected: plan.n_devices,
            got: device_buffers.len(),
        });
    }
    let buf_rows = plan.experts_per_device * plan.capacity;
    for (d, buf) in device_buffers.iter().enumerate() {
        if buf.len() != buf_rows * d_model {
            return Err(MoeError::DispatchFailed {
                msg: format!(
                    "device {d} buffer length {} != {}",
                    buf.len(),
                    buf_rows * d_model
                ),
            });
        }
    }
    if scores.is_some_and(|s| s.len() != n_tokens) {
        return Err(MoeError::DimensionMismatch {
            expected: n_tokens,
            got: scores.map_or(0, <[f32]>::len),
        });
    }

    let mut output = vec![0.0_f32; n_tokens * d_model];
    for (tok, p) in plan.placements.iter().enumerate() {
        if p.is_dropped() {
            continue;
        }
        let offset = plan.device_buffer_offset(p) * d_model;
        let src = &device_buffers[p.device][offset..offset + d_model];
        let scale = scores.map_or(1.0, |s| s[tok]);
        let dst = &mut output[tok * d_model..(tok + 1) * d_model];
        for (o, &v) in dst.iter_mut().zip(src.iter()) {
            *o += scale * v;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_experts: usize, n_devices: usize) -> ExpertParallelConfig {
        ExpertParallelConfig {
            n_experts,
            n_devices,
            capacity_factor: 2.0,
            min_capacity: 1,
        }
    }

    #[test]
    fn plan_partitions_experts_across_devices() {
        // 8 experts over 4 devices → 2 experts/device.
        let assigns = vec![0, 2, 4, 6, 1, 3, 5, 7];
        let plan = build_dispatch_plan(&assigns, 8, &cfg(8, 4)).expect("value should be present");
        assert_eq!(plan.experts_per_device, 2);
        assert_eq!(plan.n_devices, 4);
        // Expert 0 → device 0, expert 4 → device 2, expert 7 → device 3.
        assert_eq!(plan.placements[0].device, 0);
        assert_eq!(plan.placements[2].device, 2);
        assert_eq!(plan.placements[7].device, 3);
    }

    #[test]
    fn plan_local_expert_index() {
        // 4 experts / 2 devices → experts {0,1}→dev0 local {0,1}, {2,3}→dev1.
        let assigns = vec![0, 1, 2, 3];
        let plan = build_dispatch_plan(&assigns, 4, &cfg(4, 2)).expect("value should be present");
        assert_eq!(plan.placements[0].local_expert, 0);
        assert_eq!(plan.placements[1].local_expert, 1);
        assert_eq!(plan.placements[2].local_expert, 0); // expert 2 is local 0 on dev1
        assert_eq!(plan.placements[3].local_expert, 1);
    }

    #[test]
    fn plan_tokens_per_device_counts() {
        let assigns = vec![0, 0, 2, 2, 2]; // 2 to dev0, 3 to dev1
        let plan = build_dispatch_plan(&assigns, 5, &cfg(4, 2)).expect("value should be present");
        assert_eq!(plan.tokens_per_device[0], 2);
        assert_eq!(plan.tokens_per_device[1], 3);
    }

    #[test]
    fn plan_not_divisible_errors() {
        let assigns = vec![0, 1, 2];
        assert!(matches!(
            build_dispatch_plan(&assigns, 3, &cfg(5, 2)), // 5 not div by 2
            Err(MoeError::DispatchFailed { .. })
        ));
    }

    #[test]
    fn plan_zero_devices_errors() {
        let assigns = vec![0, 1];
        assert!(matches!(
            build_dispatch_plan(&assigns, 2, &cfg(4, 0)),
            Err(MoeError::InvalidExpertCount { .. })
        ));
    }

    #[test]
    fn plan_overflow_drops_tokens() {
        // 6 tokens all to expert 0; capacity = ceil(6/4 * 2)=3 → 3 dropped.
        let assigns = vec![0usize; 6];
        let plan = build_dispatch_plan(&assigns, 6, &cfg(4, 2)).expect("value should be present");
        assert_eq!(plan.capacity, 3);
        assert_eq!(plan.n_dropped, 3);
    }

    #[test]
    fn plan_dropped_assignment_passthrough() {
        let assigns = vec![0, usize::MAX, 2, usize::MAX];
        let plan = build_dispatch_plan(&assigns, 4, &cfg(4, 2)).expect("value should be present");
        assert!(plan.placements[1].is_dropped());
        assert!(plan.placements[3].is_dropped());
        assert!(!plan.placements[0].is_dropped());
    }

    #[test]
    fn plan_out_of_range_expert_errors() {
        let assigns = vec![0, 9]; // 9 >= 4
        assert!(matches!(
            build_dispatch_plan(&assigns, 2, &cfg(4, 2)),
            Err(MoeError::ExpertIndexOutOfRange { .. })
        ));
    }

    #[test]
    fn plan_empty_tokens_errors() {
        let assigns: Vec<usize> = vec![];
        assert!(matches!(
            build_dispatch_plan(&assigns, 0, &cfg(4, 2)),
            Err(MoeError::EmptyInput)
        ));
    }

    #[test]
    fn dispatch_combine_round_trip_identity() {
        // Without modifying the buffers, dispatch→combine recovers placed tokens.
        let d = 3;
        let n = 4;
        let x: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        let assigns = vec![0, 2, 1, 3];
        let plan = build_dispatch_plan(&assigns, n, &cfg(4, 2)).expect("value should be present");
        let bufs =
            dispatch_all_to_all(&x, &plan, n, d).expect("dispatch_all_to_all should succeed");
        let back = combine_all_to_all(&bufs, &plan, n, d, None)
            .expect("combine_all_to_all should succeed");
        assert_eq!(back, x, "dispatch∘combine should be identity");
    }

    #[test]
    fn dispatch_produces_per_device_buffers() {
        let d = 2;
        let n = 4;
        let x = vec![1.0_f32; n * d];
        let assigns = vec![0, 1, 2, 3];
        let plan = build_dispatch_plan(&assigns, n, &cfg(4, 2)).expect("value should be present");
        let bufs =
            dispatch_all_to_all(&x, &plan, n, d).expect("dispatch_all_to_all should succeed");
        assert_eq!(bufs.len(), 2); // one per device
        let buf_rows = plan.experts_per_device * plan.capacity;
        assert_eq!(bufs[0].len(), buf_rows * d);
    }

    #[test]
    fn dispatch_wrong_x_size_errors() {
        let assigns = vec![0, 1];
        let plan = build_dispatch_plan(&assigns, 2, &cfg(4, 2)).expect("value should be present");
        let x = vec![0.0_f32; 5]; // not 2*d
        assert!(matches!(
            dispatch_all_to_all(&x, &plan, 2, 3),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn combine_applies_scores() {
        let d = 2;
        let n = 2;
        let x = vec![1.0_f32, 1.0, 1.0, 1.0];
        let assigns = vec![0, 2];
        let plan = build_dispatch_plan(&assigns, n, &cfg(4, 2)).expect("value should be present");
        let bufs =
            dispatch_all_to_all(&x, &plan, n, d).expect("dispatch_all_to_all should succeed");
        let scores = vec![3.0_f32, 0.25];
        let out =
            combine_all_to_all(&bufs, &plan, n, d, Some(&scores)).expect("value should be present");
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn combine_wrong_buffer_count_errors() {
        let assigns = vec![0, 2];
        let plan = build_dispatch_plan(&assigns, 2, &cfg(4, 2)).expect("value should be present");
        let bufs = vec![vec![0.0_f32; 10]]; // only 1 buffer, need 2
        assert!(matches!(
            combine_all_to_all(&bufs, &plan, 2, 2, None),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn combine_dropped_token_zero_output() {
        let d = 2;
        let n = 6;
        let x = vec![1.0_f32; n * d];
        let assigns = vec![0usize; n]; // overflow drops some
        let plan = build_dispatch_plan(&assigns, n, &cfg(4, 2)).expect("value should be present");
        let bufs =
            dispatch_all_to_all(&x, &plan, n, d).expect("dispatch_all_to_all should succeed");
        let out = combine_all_to_all(&bufs, &plan, n, d, None)
            .expect("combine_all_to_all should succeed");
        // The dropped tokens must have an all-zero output row.
        let n_zero_rows = (0..n)
            .filter(|&t| out[t * d..(t + 1) * d].iter().all(|&v| v == 0.0))
            .count();
        assert_eq!(n_zero_rows, plan.n_dropped);
    }

    #[test]
    fn device_buffer_offset_layout() {
        let assigns = vec![0, 1, 2, 3];
        let plan = build_dispatch_plan(&assigns, 4, &cfg(4, 2)).expect("value should be present");
        let p = TokenPlacement {
            device: 0,
            local_expert: 1,
            slot: 0,
        };
        assert_eq!(plan.device_buffer_offset(&p), plan.capacity);
    }

    #[test]
    fn single_device_is_plain_capacity() {
        // n_devices = 1 → every expert local; behaves like one big buffer.
        let assigns = vec![0, 1, 2, 3];
        let plan = build_dispatch_plan(&assigns, 4, &cfg(4, 1)).expect("value should be present");
        assert_eq!(plan.n_devices, 1);
        assert_eq!(plan.experts_per_device, 4);
        assert!(plan.placements.iter().all(|p| p.device == 0));
    }
}
