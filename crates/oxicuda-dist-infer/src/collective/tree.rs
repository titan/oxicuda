//! Tree / recursive collective step schedules (MPICH-style).
//!
//! For *small* messages the ring's `2·(n−1)` step latency is dominated by the
//! per-step handshake, so latency-optimal `log₂ n`-step algorithms win:
//!
//! * **Recursive halving** all-reduce — each rank pairs with a partner at
//!   distance `n/2, n/4, …, 1`; halves are exchanged & reduced so that after
//!   `log₂ n` steps every rank holds the full sum (this implementation uses the
//!   *recursive-distance-doubling* form, which is `log₂ n` steps and reduces
//!   the whole vector each step — exact and simple to verify).
//! * **Recursive doubling** all-gather — partners at distance `1, 2, 4, …`
//!   exchange the data they currently hold; after `log₂ n` steps every rank has
//!   gathered all contributions.
//!
//! Both require `world_size` to be a power of two (the classic MPICH
//! constraint; non-power-of-two needs the "extra ranks" pre/post step, omitted
//! here for clarity). The in-memory executors are bit-exact oracles.
//!
//! # Reference
//! - Thakur, Rabenseifner, Gropp (2005) "Optimization of Collective
//!   Communication Operations in MPICH." IJHPCA.

use crate::error::{DistInferError, DistInferResult};

/// A single pairwise exchange in a recursive (tree) schedule.
///
/// On this step `rank` exchanges with `partner = rank XOR (1 << step)`. Both
/// directions of the exchange are emitted (one [`TreeStep`] per ordered pair),
/// so the schedule lists `world_size` entries per round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeStep {
    /// Round index `0 .. log2(world_size)`.
    pub step: usize,
    /// Rank initiating this directed transfer.
    pub rank: usize,
    /// Exchange partner = `rank XOR (1 << step)`.
    pub partner: usize,
    /// Distance between the pair = `1 << step`.
    pub distance: usize,
}

/// Returns `Some(log2(n))` iff `n` is a power of two and ≥ 1.
fn log2_pow2(n: usize) -> Option<u32> {
    if n == 0 || (n & (n - 1)) != 0 {
        None
    } else {
        Some(n.trailing_zeros())
    }
}

/// Generate the recursive-halving (distance-doubling) **all-reduce** schedule.
///
/// `world_size` must be a power of two. Produces `log₂(n) · n` directed
/// exchanges (one per rank per round). On round `s` every rank exchanges its
/// current full vector with `rank XOR (1 << s)` and reduces.
///
/// # Errors
///
/// [`DistInferError::InvalidWorldSize`] if `world_size` is not a power of two.
pub fn recursive_halving_all_reduce_schedule(world_size: usize) -> DistInferResult<Vec<TreeStep>> {
    let rounds = log2_pow2(world_size).ok_or(DistInferError::InvalidWorldSize {
        world_size,
        reason: "recursive-halving requires power-of-two world size",
    })?;
    let mut steps = Vec::with_capacity(rounds as usize * world_size);
    for step in 0..rounds as usize {
        let distance = 1usize << step;
        for rank in 0..world_size {
            steps.push(TreeStep {
                step,
                rank,
                partner: rank ^ distance,
                distance,
            });
        }
    }
    Ok(steps)
}

/// Generate the recursive-doubling **all-gather** schedule.
///
/// Identical pairing structure to the all-reduce, but the data each rank holds
/// *grows* each round instead of being reduced. `world_size` must be a power of
/// two.
///
/// # Errors
///
/// [`DistInferError::InvalidWorldSize`] if `world_size` is not a power of two.
pub fn recursive_doubling_all_gather_schedule(world_size: usize) -> DistInferResult<Vec<TreeStep>> {
    // Same pairing as halving all-reduce.
    recursive_halving_all_reduce_schedule(world_size)
}

/// Execute the recursive-halving **all-reduce** over per-rank buffers.
///
/// Every rank contributes a vector of equal length; the returned vector equals
/// the exact element-wise sum that every rank would hold. The executor follows
/// the pairwise schedule (it does not just sum), validating the schedule.
///
/// # Errors
///
/// * [`DistInferError::InvalidWorldSize`] if `inputs.len()` is not a power of
///   two.
/// * [`DistInferError::DimensionMismatch`] if input lengths differ or the
///   schedule fails to converge.
pub fn execute_recursive_halving_all_reduce(inputs: &[Vec<f32>]) -> DistInferResult<Vec<f32>> {
    let n = inputs.len();
    let rounds = log2_pow2(n).ok_or(DistInferError::InvalidWorldSize {
        world_size: n,
        reason: "recursive-halving requires power-of-two world size",
    })?;
    let len = inputs.first().map_or(0, Vec::len);
    for inp in inputs {
        if inp.len() != len {
            return Err(DistInferError::DimensionMismatch {
                expected: len,
                got: inp.len(),
            });
        }
    }

    let mut buf: Vec<Vec<f32>> = inputs.to_vec();
    for step in 0..rounds as usize {
        let distance = 1usize << step;
        // Snapshot all sends before any receive so the round is atomic.
        let snapshot = buf.clone();
        for (rank, rank_buf) in buf.iter_mut().enumerate() {
            let partner = rank ^ distance;
            for (d, &s) in rank_buf.iter_mut().zip(snapshot[partner].iter()) {
                *d += s;
            }
        }
    }

    // Oracle.
    let mut oracle = vec![0.0_f32; len];
    for inp in inputs {
        for (o, &v) in oracle.iter_mut().zip(inp.iter()) {
            *o += v;
        }
    }
    for rank_buf in &buf {
        for (&got, &exp) in rank_buf.iter().zip(oracle.iter()) {
            if (got - exp).abs() > 1e-3 * (1.0 + exp.abs()) {
                return Err(DistInferError::Internal(
                    "recursive-halving all-reduce did not converge to the exact sum",
                ));
            }
        }
    }
    Ok(buf.into_iter().next().unwrap_or_default())
}

/// Execute the recursive-doubling **all-gather** over per-rank chunks.
///
/// `chunks[r]` is rank `r`'s contribution of equal length `chunk_len`; the
/// returned buffer is the rank-ordered concatenation
/// (`world_size · chunk_len`) that every rank ends up holding.
///
/// # Errors
///
/// * [`DistInferError::InvalidWorldSize`] if `chunks.len()` is not a power of
///   two.
/// * [`DistInferError::DimensionMismatch`] on unequal chunk lengths or
///   non-convergence.
pub fn execute_recursive_doubling_all_gather(chunks: &[Vec<f32>]) -> DistInferResult<Vec<f32>> {
    let n = chunks.len();
    let rounds = log2_pow2(n).ok_or(DistInferError::InvalidWorldSize {
        world_size: n,
        reason: "recursive-doubling requires power-of-two world size",
    })?;
    let cl = chunks.first().map_or(0, Vec::len);
    for c in chunks {
        if c.len() != cl {
            return Err(DistInferError::DimensionMismatch {
                expected: cl,
                got: c.len(),
            });
        }
    }

    // Each rank's working set: slot c holds chunk c's data once known.
    let mut buf: Vec<Vec<Option<Vec<f32>>>> = (0..n)
        .map(|r| {
            (0..n)
                .map(|c| {
                    if c == r {
                        Some(chunks[r].clone())
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    for step in 0..rounds as usize {
        let distance = 1usize << step;
        let snapshot = buf.clone();
        for (rank, rank_buf) in buf.iter_mut().enumerate() {
            let partner = rank ^ distance;
            for (slot, src) in rank_buf.iter_mut().zip(snapshot[partner].iter()) {
                if slot.is_none() {
                    if let Some(data) = src {
                        *slot = Some(data.clone());
                    }
                }
            }
        }
    }

    let mut result = vec![0.0_f32; n * cl];
    for (c, slot) in buf[0].iter().enumerate() {
        let slot = slot
            .as_ref()
            .ok_or(DistInferError::Internal("all-gather left a hole"))?;
        result[c * cl..(c + 1) * cl].copy_from_slice(slot);
    }
    // Validate every rank fully gathered.
    for rank_buf in &buf {
        if rank_buf.iter().any(Option::is_none) {
            return Err(DistInferError::Internal(
                "recursive-doubling all-gather did not reach every rank",
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

    #[test]
    fn log2_pow2_classifies() {
        assert_eq!(log2_pow2(1), Some(0));
        assert_eq!(log2_pow2(2), Some(1));
        assert_eq!(log2_pow2(8), Some(3));
        assert_eq!(log2_pow2(0), None);
        assert_eq!(log2_pow2(6), None);
    }

    #[test]
    fn halving_schedule_rounds_and_pairs() {
        let sched = recursive_halving_all_reduce_schedule(8).expect("sched");
        // log2(8)=3 rounds × 8 ranks.
        assert_eq!(sched.len(), 3 * 8);
        for st in &sched {
            assert_eq!(st.partner, st.rank ^ st.distance);
            assert_eq!(st.distance, 1 << st.step);
        }
    }

    #[test]
    fn halving_non_pow2_errors() {
        assert!(matches!(
            recursive_halving_all_reduce_schedule(6),
            Err(DistInferError::InvalidWorldSize { .. })
        ));
    }

    #[test]
    fn all_reduce_uniform_exact() {
        let inputs = vec![vec![1.0_f32; 4]; 8];
        let out = execute_recursive_halving_all_reduce(&inputs).expect("ar");
        assert_eq!(out, vec![8.0_f32; 4]);
    }

    #[test]
    fn all_reduce_ramp_exact() {
        let inputs: Vec<Vec<f32>> = (0..4)
            .map(|r| (0..3).map(|i| (r * 10 + i) as f32).collect())
            .collect();
        let out = execute_recursive_halving_all_reduce(&inputs).expect("ar");
        let mut exp = vec![0.0_f32; 3];
        for inp in &inputs {
            for (e, &v) in exp.iter_mut().zip(inp.iter()) {
                *e += v;
            }
        }
        assert_eq!(out, exp);
    }

    #[test]
    fn all_reduce_random_matches_oracle() {
        let mut rng = DistInferRng::new(11);
        let n = 8;
        let len = 7;
        let inputs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..len).map(|_| rng.next_gaussian()).collect())
            .collect();
        let out = execute_recursive_halving_all_reduce(&inputs).expect("ar");
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
    fn all_reduce_non_pow2_errors() {
        let inputs = vec![vec![1.0_f32; 4]; 3];
        assert!(matches!(
            execute_recursive_halving_all_reduce(&inputs),
            Err(DistInferError::InvalidWorldSize { .. })
        ));
    }

    #[test]
    fn all_gather_reconstructs() {
        let n = 4;
        let cl = 2;
        let chunks: Vec<Vec<f32>> = (0..n)
            .map(|r| (0..cl).map(|i| (r * 10 + i) as f32).collect())
            .collect();
        let full = execute_recursive_doubling_all_gather(&chunks).expect("ag");
        let mut exp = vec![0.0_f32; n * cl];
        for (r, c) in chunks.iter().enumerate() {
            exp[r * cl..(r + 1) * cl].copy_from_slice(c);
        }
        assert_eq!(full, exp);
    }

    #[test]
    fn all_gather_eight_ranks() {
        let n = 8;
        let cl = 3;
        let chunks: Vec<Vec<f32>> = (0..n)
            .map(|r| (0..cl).map(|i| (r * 100 + i) as f32).collect())
            .collect();
        let full = execute_recursive_doubling_all_gather(&chunks).expect("ag");
        assert_eq!(full.len(), n * cl);
        for (r, c) in chunks.iter().enumerate() {
            assert_eq!(&full[r * cl..(r + 1) * cl], &c[..]);
        }
    }

    #[test]
    fn all_gather_non_pow2_errors() {
        let chunks = vec![vec![1.0_f32; 2]; 3];
        assert!(matches!(
            execute_recursive_doubling_all_gather(&chunks),
            Err(DistInferError::InvalidWorldSize { .. })
        ));
    }

    #[test]
    fn single_rank_identity() {
        let inputs = vec![vec![5.0_f32, 6.0, 7.0]];
        let out = execute_recursive_halving_all_reduce(&inputs).expect("ar");
        assert_eq!(out, vec![5.0, 6.0, 7.0]);
        let full = execute_recursive_doubling_all_gather(&inputs).expect("ag");
        assert_eq!(full, vec![5.0, 6.0, 7.0]);
    }
}
