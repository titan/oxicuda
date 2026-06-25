//! Ring collective step schedules (Baidu / NCCL ring algorithm).
//!
//! On a ring of `world_size = n` ranks, a tensor is split into `n` equal
//! *chunks*. Rank `r` sends to `(r + 1) % n` and receives from
//! `(r + n − 1) % n`. The classic bandwidth-optimal **ring all-reduce** is two
//! phases, each `n − 1` steps:
//!
//! 1. **Reduce-scatter** — after `n − 1` steps, rank `r` owns the fully reduced
//!    value of chunk `(r + 1) % n` (the chunk it will be responsible for during
//!    all-gather), having accumulated one peer contribution per step.
//! 2. **All-gather** — the reduced chunks circulate so that, after another
//!    `n − 1` steps, every rank holds every reduced chunk.
//!
//! Total data moved per rank is `2·(n − 1)/n · |tensor|` — independent of `n`
//! to first order, which is why the ring is bandwidth-optimal.
//!
//! This module emits the **step schedule** (which chunk each rank ships to its
//! successor on each step) and provides **in-memory executors** that run the
//! schedule over host buffers. The executor output is the exact element-wise
//! sum / concatenation, giving a bit-exact oracle for the device path.

use crate::error::{DistInferError, DistInferResult};

// ─── RingStep ────────────────────────────────────────────────────────────────

/// A single point-to-point transfer in a ring schedule.
///
/// On this step, `src_rank` sends its copy of chunk `chunk` to `dst_rank`
/// (`dst_rank == (src_rank + 1) % world_size`). The receiver either *adds* the
/// incoming chunk into its accumulator (reduce-scatter phase) or *overwrites*
/// its slot (all-gather phase), as recorded by [`RingStep::reduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingStep {
    /// Zero-based step index within the phase.
    pub step: usize,
    /// Rank transmitting on this step.
    pub src_rank: usize,
    /// Rank receiving on this step (always `src_rank`'s ring successor).
    pub dst_rank: usize,
    /// Index of the chunk being transmitted.
    pub chunk: usize,
    /// `true` → receiver accumulates (reduce-scatter); `false` → receiver
    /// copies (all-gather).
    pub reduce: bool,
}

// ─── RingCollective ──────────────────────────────────────────────────────────

/// Describes a ring of `world_size` ranks and the chunking of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingCollective {
    /// Number of ranks on the ring.
    pub world_size: usize,
    /// Length (in elements) of the tensor each rank contributes.
    pub tensor_len: usize,
}

impl RingCollective {
    /// Construct a ring descriptor.
    ///
    /// # Errors
    ///
    /// * [`DistInferError::TooFewRanks`] if `world_size < 1`.
    /// * [`DistInferError::DimensionMismatch`] if `tensor_len` is not divisible
    ///   by `world_size` (the ring algorithm requires equal chunks).
    pub fn new(world_size: usize, tensor_len: usize) -> DistInferResult<Self> {
        if world_size == 0 {
            return Err(DistInferError::TooFewRanks {
                needed: 1,
                world_size,
            });
        }
        if tensor_len % world_size != 0 {
            return Err(DistInferError::DimensionMismatch {
                expected: world_size,
                got: tensor_len,
            });
        }
        Ok(Self {
            world_size,
            tensor_len,
        })
    }

    /// Length of one chunk = `tensor_len / world_size`.
    #[must_use]
    pub fn chunk_len(&self) -> usize {
        self.tensor_len / self.world_size
    }

    /// The ring successor of `rank`.
    #[must_use]
    pub fn successor(&self, rank: usize) -> usize {
        (rank + 1) % self.world_size
    }

    /// The ring predecessor of `rank`.
    #[must_use]
    pub fn predecessor(&self, rank: usize) -> usize {
        (rank + self.world_size - 1) % self.world_size
    }
}

// ─── Schedule generation ───────────────────────────────────────────────────────

/// Generate the **reduce-scatter** phase of a ring all-reduce.
///
/// Produces `world_size · (world_size − 1)` steps (across all ranks). On
/// step `s` rank `r` sends chunk `(r − s + world_size) % world_size`; the
/// receiver accumulates it. After all steps, rank `r` holds the fully reduced
/// value of chunk `(r + 1) % world_size`.
#[must_use]
pub fn ring_reduce_scatter_schedule(ring: &RingCollective) -> Vec<RingStep> {
    let n = ring.world_size;
    let mut steps = Vec::with_capacity(n.saturating_sub(1) * n);
    for step in 0..n.saturating_sub(1) {
        for src in 0..n {
            // Standard NCCL ring reduce-scatter: on step `s`, rank `r` sends
            // the chunk it most-recently accumulated, indexed `(r - s) mod n`.
            let chunk = (src + n - (step % n)) % n;
            steps.push(RingStep {
                step,
                src_rank: src,
                dst_rank: (src + 1) % n,
                chunk,
                reduce: true,
            });
        }
    }
    steps
}

/// Generate the **all-gather** phase of a ring all-reduce / a standalone ring
/// all-gather.
///
/// After reduce-scatter, rank `r` owns reduced chunk `(r + 1) % n`. This phase
/// circulates owned chunks so every rank ends with every chunk. On step `s`
/// rank `r` forwards chunk `(r + 1 − s + n) % n`; the receiver overwrites.
#[must_use]
pub fn ring_all_gather_schedule(ring: &RingCollective) -> Vec<RingStep> {
    let n = ring.world_size;
    let mut steps = Vec::with_capacity(n.saturating_sub(1) * n);
    for step in 0..n.saturating_sub(1) {
        for src in 0..n {
            // After reduce-scatter rank `r` owns chunk (r+1) mod n; forward the
            // chunk it received on the previous step.
            let chunk = (src + 1 + n - (step % n)) % n;
            steps.push(RingStep {
                step,
                src_rank: src,
                dst_rank: (src + 1) % n,
                chunk,
                reduce: false,
            });
        }
    }
    steps
}

/// Generate the full ring **all-reduce** schedule = reduce-scatter then
/// all-gather. The two phases are concatenated; phase boundary is at
/// `world_size − 1` distinct step indices each.
#[must_use]
pub fn ring_all_reduce_schedule(ring: &RingCollective) -> Vec<RingStep> {
    let mut sched = ring_reduce_scatter_schedule(ring);
    sched.extend(ring_all_gather_schedule(ring));
    sched
}

// ─── In-memory executors (the oracle) ──────────────────────────────────────────

/// Execute a ring **all-reduce** over per-rank input buffers, returning the
/// fully reduced buffer that every rank would hold.
///
/// `inputs[r]` is rank `r`'s contribution, each of length `tensor_len`. The
/// returned buffer equals the exact element-wise sum across all ranks — this is
/// the oracle for verifying a real device ring all-reduce.
///
/// The executor faithfully *simulates the ring data motion* (it does not simply
/// sum the inputs): it allocates per-rank chunk slots, runs the reduce-scatter
/// then all-gather schedules, and asserts every rank converges to the same
/// result. A mismatch indicates a bug in the generated schedule.
///
/// # Errors
///
/// * [`DistInferError::TooFewRanks`] if `inputs` is empty.
/// * [`DistInferError::DimensionMismatch`] if any input length disagrees with
///   `world_size · chunk_len`, or if the schedule fails to converge.
pub fn execute_ring_all_reduce(
    ring: &RingCollective,
    inputs: &[Vec<f32>],
) -> DistInferResult<Vec<f32>> {
    let n = ring.world_size;
    if inputs.len() != n {
        return Err(DistInferError::DimensionMismatch {
            expected: n,
            got: inputs.len(),
        });
    }
    let cl = ring.chunk_len();
    for inp in inputs {
        if inp.len() != ring.tensor_len {
            return Err(DistInferError::DimensionMismatch {
                expected: ring.tensor_len,
                got: inp.len(),
            });
        }
    }

    // Per-rank working buffers: buf[r][chunk] is rank r's copy of chunk's data.
    // Initialise from each rank's own contribution.
    let mut buf: Vec<Vec<Vec<f32>>> = inputs
        .iter()
        .map(|inp| (0..n).map(|c| inp[c * cl..(c + 1) * cl].to_vec()).collect())
        .collect();

    // ── Reduce-scatter: accumulate one peer chunk per step ────────────────────
    // Snapshot of what each rank will SEND on a step must be taken before any
    // receiver mutates, so transfers within a step are independent.
    for step in ring_reduce_scatter_schedule(ring) {
        let payload = buf[step.src_rank][step.chunk].clone();
        let dst = &mut buf[step.dst_rank][step.chunk];
        for (d, s) in dst.iter_mut().zip(payload.iter()) {
            *d += *s;
        }
    }

    // After reduce-scatter, rank r holds the fully reduced value of chunk
    // (r + 1) mod n. Run all-gather to propagate it to all ranks.
    for step in ring_all_gather_schedule(ring) {
        let payload = buf[step.src_rank][step.chunk].clone();
        buf[step.dst_rank][step.chunk].copy_from_slice(&payload);
    }

    // Compute the exact oracle independently to validate convergence.
    let mut oracle = vec![0.0_f32; ring.tensor_len];
    for inp in inputs {
        for (o, &v) in oracle.iter_mut().zip(inp.iter()) {
            *o += v;
        }
    }

    // Every rank must now hold `oracle`; flatten rank 0 and verify all match.
    let mut result = Vec::with_capacity(ring.tensor_len);
    for chunk in &buf[0] {
        result.extend_from_slice(chunk);
    }
    for rank_buf in &buf {
        let mut flat = Vec::with_capacity(ring.tensor_len);
        for chunk in rank_buf {
            flat.extend_from_slice(chunk);
        }
        for (&got, &exp) in flat.iter().zip(oracle.iter()) {
            if (got - exp).abs() > 1e-3 * (1.0 + exp.abs()) {
                return Err(DistInferError::Internal(
                    "ring all-reduce schedule did not converge to the exact sum",
                ));
            }
        }
    }
    Ok(result)
}

/// Execute the ring **reduce-scatter** alone: returns, per rank, the fully
/// reduced chunk that rank owns afterwards.
///
/// `out[r]` has length `chunk_len` and equals the element-wise sum (across all
/// ranks) of chunk `(r + 1) % world_size`.
///
/// # Errors
///
/// As [`execute_ring_all_reduce`].
pub fn execute_ring_reduce_scatter(
    ring: &RingCollective,
    inputs: &[Vec<f32>],
) -> DistInferResult<Vec<Vec<f32>>> {
    let n = ring.world_size;
    if inputs.len() != n {
        return Err(DistInferError::DimensionMismatch {
            expected: n,
            got: inputs.len(),
        });
    }
    let cl = ring.chunk_len();
    for inp in inputs {
        if inp.len() != ring.tensor_len {
            return Err(DistInferError::DimensionMismatch {
                expected: ring.tensor_len,
                got: inp.len(),
            });
        }
    }

    let mut buf: Vec<Vec<Vec<f32>>> = inputs
        .iter()
        .map(|inp| (0..n).map(|c| inp[c * cl..(c + 1) * cl].to_vec()).collect())
        .collect();

    for step in ring_reduce_scatter_schedule(ring) {
        let payload = buf[step.src_rank][step.chunk].clone();
        let dst = &mut buf[step.dst_rank][step.chunk];
        for (d, s) in dst.iter_mut().zip(payload.iter()) {
            *d += *s;
        }
    }

    // Rank r owns reduced chunk (r + 1) mod n.
    let owned: Vec<Vec<f32>> = (0..n).map(|r| buf[r][(r + 1) % n].clone()).collect();
    Ok(owned)
}

/// Execute a ring **all-gather**: given each rank's owned chunk, returns the
/// concatenation every rank ends up holding (`world_size · chunk_len`).
///
/// `owned[r]` is rank `r`'s chunk; the result places chunk `r` at offset
/// `r · chunk_len` in the output (rank order).
///
/// # Errors
///
/// * [`DistInferError::TooFewRanks`] if `owned` is empty.
/// * [`DistInferError::DimensionMismatch`] on a chunk-length mismatch.
pub fn execute_ring_all_gather(
    ring: &RingCollective,
    owned: &[Vec<f32>],
) -> DistInferResult<Vec<f32>> {
    let n = ring.world_size;
    if owned.len() != n {
        return Err(DistInferError::DimensionMismatch {
            expected: n,
            got: owned.len(),
        });
    }
    let cl = ring.chunk_len();
    for chunk in owned {
        if chunk.len() != cl {
            return Err(DistInferError::DimensionMismatch {
                expected: cl,
                got: chunk.len(),
            });
        }
    }

    // Per-rank slot table; rank r starts owning chunk r only (NCCL all-gather
    // convention: chunk index == owning rank).
    let mut buf: Vec<Vec<Option<Vec<f32>>>> = (0..n)
        .map(|r| {
            (0..n)
                .map(|c| if c == r { Some(owned[r].clone()) } else { None })
                .collect()
        })
        .collect();

    // Ring all-gather: each rank forwards the chunk it just received. On step
    // `s`, rank `r` forwards chunk `(r − s + n) % n`.
    for step in 0..n.saturating_sub(1) {
        // Snapshot sends first so a step's transfers are independent.
        let mut pending: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(n);
        for (r, rank_buf) in buf.iter().enumerate() {
            let chunk = (r + n - (step % n)) % n;
            if let Some(data) = &rank_buf[chunk] {
                pending.push(((r + 1) % n, chunk, data.clone()));
            }
        }
        for (dst, chunk, data) in pending {
            buf[dst][chunk] = Some(data);
        }
    }

    let mut result = vec![0.0_f32; ring.tensor_len];
    for (c, slot) in buf[0].iter().enumerate() {
        let slot = slot
            .as_ref()
            .ok_or(DistInferError::Internal("ring all-gather left a hole"))?;
        result[c * cl..(c + 1) * cl].copy_from_slice(slot);
    }
    // Validate every rank converged.
    for rank_buf in &buf {
        if rank_buf.iter().any(Option::is_none) {
            return Err(DistInferError::Internal(
                "ring all-gather did not reach every rank",
            ));
        }
    }
    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::DistInferRng;

    fn ramp(rank: usize, len: usize) -> Vec<f32> {
        (0..len).map(|i| (rank * 100 + i) as f32).collect()
    }

    #[test]
    fn ring_construct_validates_divisibility() {
        assert!(RingCollective::new(4, 16).is_ok());
        assert!(matches!(
            RingCollective::new(3, 16),
            Err(DistInferError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            RingCollective::new(0, 16),
            Err(DistInferError::TooFewRanks { .. })
        ));
    }

    #[test]
    fn ring_successor_predecessor() {
        let ring = RingCollective::new(4, 8).expect("ring");
        assert_eq!(ring.successor(3), 0);
        assert_eq!(ring.predecessor(0), 3);
        assert_eq!(ring.chunk_len(), 2);
    }

    #[test]
    fn reduce_scatter_has_n_minus_1_steps_per_rank() {
        let ring = RingCollective::new(4, 16).expect("ring");
        let sched = ring_reduce_scatter_schedule(&ring);
        // (n-1) steps × n ranks.
        assert_eq!(sched.len(), 3 * 4);
        // Every send is to the immediate successor.
        for st in &sched {
            assert_eq!(st.dst_rank, (st.src_rank + 1) % 4);
            assert!(st.reduce);
        }
    }

    #[test]
    fn all_reduce_full_schedule_step_count() {
        let ring = RingCollective::new(5, 20).expect("ring");
        let sched = ring_all_reduce_schedule(&ring);
        // 2 phases × (n-1) × n.
        assert_eq!(sched.len(), 2 * 4 * 5);
    }

    #[test]
    fn all_reduce_yields_exact_sum_uniform() {
        let ring = RingCollective::new(4, 16).expect("ring");
        let inputs: Vec<Vec<f32>> = (0..4).map(|_| vec![1.0_f32; 16]).collect();
        let out = execute_ring_all_reduce(&ring, &inputs).expect("all-reduce");
        assert_eq!(out, vec![4.0_f32; 16]);
    }

    #[test]
    fn all_reduce_yields_exact_sum_ramp() {
        let ring = RingCollective::new(4, 16).expect("ring");
        let inputs: Vec<Vec<f32>> = (0..4).map(|r| ramp(r, 16)).collect();
        let out = execute_ring_all_reduce(&ring, &inputs).expect("all-reduce");
        // Oracle.
        let mut exp = vec![0.0_f32; 16];
        for inp in &inputs {
            for (e, &v) in exp.iter_mut().zip(inp.iter()) {
                *e += v;
            }
        }
        assert_eq!(out, exp);
    }

    #[test]
    fn all_reduce_single_rank_is_identity() {
        let ring = RingCollective::new(1, 5).expect("ring");
        let inputs = vec![vec![3.0_f32, 1.0, 4.0, 1.0, 5.0]];
        let out = execute_ring_all_reduce(&ring, &inputs).expect("all-reduce");
        assert_eq!(out, inputs[0]);
    }

    #[test]
    fn all_reduce_random_matches_oracle() {
        let mut rng = DistInferRng::new(7);
        let n = 8;
        let len = 8 * 6; // 48
        let ring = RingCollective::new(n, len).expect("ring");
        let inputs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..len).map(|_| rng.next_gaussian()).collect())
            .collect();
        let out = execute_ring_all_reduce(&ring, &inputs).expect("all-reduce");
        let mut exp = vec![0.0_f32; len];
        for inp in &inputs {
            for (e, &v) in exp.iter_mut().zip(inp.iter()) {
                *e += v;
            }
        }
        for (g, e) in out.iter().zip(exp.iter()) {
            assert!((g - e).abs() < 1e-3 * (1.0 + e.abs()), "got {g} exp {e}");
        }
    }

    #[test]
    fn reduce_scatter_owned_chunk_is_reduced() {
        let ring = RingCollective::new(4, 16).expect("ring");
        let inputs: Vec<Vec<f32>> = (0..4).map(|r| ramp(r, 16)).collect();
        let owned = execute_ring_reduce_scatter(&ring, &inputs).expect("rs");
        let cl = ring.chunk_len();
        // Rank r owns reduced chunk (r+1) mod n.
        for (r, owned_chunk) in owned.iter().enumerate() {
            let chunk = (r + 1) % 4;
            let mut exp = vec![0.0_f32; cl];
            for inp in &inputs {
                for (e, &v) in exp.iter_mut().zip(inp[chunk * cl..(chunk + 1) * cl].iter()) {
                    *e += v;
                }
            }
            assert_eq!(*owned_chunk, exp, "rank {r} owns wrong reduced chunk");
        }
    }

    #[test]
    fn all_gather_reconstructs_full_buffer() {
        let ring = RingCollective::new(4, 12).expect("ring");
        let cl = ring.chunk_len(); // 3
        // Rank r owns chunk r = [r*10 .. r*10+cl).
        let owned: Vec<Vec<f32>> = (0..4)
            .map(|r| (0..cl).map(|i| (r * 10 + i) as f32).collect())
            .collect();
        let full = execute_ring_all_gather(&ring, &owned).expect("ag");
        let mut exp = vec![0.0_f32; 12];
        for (r, chunk) in owned.iter().enumerate() {
            exp[r * cl..(r + 1) * cl].copy_from_slice(chunk);
        }
        assert_eq!(full, exp);
    }

    #[test]
    fn reduce_scatter_then_all_gather_equals_all_reduce() {
        let ring = RingCollective::new(4, 16).expect("ring");
        let n = ring.world_size;
        let inputs: Vec<Vec<f32>> = (0..4).map(|r| ramp(r, 16)).collect();
        // After ring reduce-scatter, rank r owns reduced chunk (r+1) mod n.
        let owned_by_rank = execute_ring_reduce_scatter(&ring, &inputs).expect("rs");
        // The all-gather contract is `owned[c]` == reduced chunk `c`; remap the
        // rank-indexed reduce-scatter output into chunk-id order to compose them
        // (chunk c is owned by rank (c - 1 + n) mod n).
        let owned_by_chunk: Vec<Vec<f32>> = (0..n)
            .map(|c| owned_by_rank[(c + n - 1) % n].clone())
            .collect();
        let full = execute_ring_all_gather(&ring, &owned_by_chunk).expect("ag");
        let direct = execute_ring_all_reduce(&ring, &inputs).expect("ar");
        assert_eq!(full, direct, "RS + AG must equal AR");
    }

    #[test]
    fn input_count_mismatch_errors() {
        let ring = RingCollective::new(4, 8).expect("ring");
        let inputs = vec![vec![1.0_f32; 8]; 3]; // only 3 of 4
        assert!(matches!(
            execute_ring_all_reduce(&ring, &inputs),
            Err(DistInferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn input_len_mismatch_errors() {
        let ring = RingCollective::new(2, 8).expect("ring");
        let inputs = vec![vec![1.0_f32; 8], vec![1.0_f32; 7]];
        assert!(matches!(
            execute_ring_all_reduce(&ring, &inputs),
            Err(DistInferError::DimensionMismatch { .. })
        ));
    }
}
