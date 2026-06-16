//! Specialised encodings for genetic algorithms.
//!
//! Provides:
//! - [`GrayEncoder`] — Gray-coded binary encoding for continuous variables.
//! - [`pmx_crossover`] — Partially Mapped Crossover (PMX) for permutation genomes.
//! - [`ox_crossover`]  — Order Crossover (OX) for permutation genomes.
//! - [`cx_crossover`]  — Cycle Crossover (CX) for permutation genomes.
//! - [`inversion_mutation`] — Reversal of a random sub-segment of a permutation.
//! - [`two_opt_improve`]   — Single-pass 2-opt local search for TSP-style tours.

use crate::{EvolError, EvolResult, handle::LcgRng};

// ─────────────────────────────────────────────────────────────────────────────
// Gray-coded binary encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Gray-coded binary encoder/decoder for unsigned integer values.
///
/// Gray code is a reflected binary code where adjacent values differ by exactly
/// one bit, which improves the fitness landscape for binary-encoded GAs by reducing
/// the Hamming distance between neighbouring phenotypes.
///
/// # Example
/// ```
/// use oxicuda_evol::GrayEncoder;
/// let enc = GrayEncoder::new(8).expect("new should succeed");
/// let gray = enc.encode(42);
/// assert_eq!(enc.decode(&gray), 42);
/// ```
#[derive(Debug, Clone)]
pub struct GrayEncoder {
    n_bits: usize,
    max_value: u64,
}

impl GrayEncoder {
    /// Create a new `GrayEncoder` for `n_bits`-wide Gray codes.
    ///
    /// # Errors
    /// Returns `EvolError::InvalidParameter` if `n_bits == 0` or `n_bits > 63`.
    pub fn new(n_bits: usize) -> EvolResult<Self> {
        if n_bits == 0 {
            return Err(EvolError::InvalidParameter(
                "GrayEncoder: n_bits must be >= 1".to_owned(),
            ));
        }
        if n_bits > 63 {
            return Err(EvolError::InvalidParameter(
                "GrayEncoder: n_bits must be <= 63 to avoid u64 overflow".to_owned(),
            ));
        }
        let max_value = (1u64 << n_bits) - 1;
        Ok(Self { n_bits, max_value })
    }

    /// Return the number of bits this encoder uses.
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    /// Encode a binary integer as a Gray code.
    ///
    /// Uses the standard XOR-shift formula: `gray = value ^ (value >> 1)`.
    /// Values exceeding the maximum representable unsigned integer are masked to
    /// `n_bits` bits before encoding.
    ///
    /// Returns a `Vec<bool>` of length `n_bits`, MSB first.
    pub fn encode(&self, value: u64) -> Vec<bool> {
        let masked = value & self.max_value;
        let gray_int = masked ^ (masked >> 1);
        (0..self.n_bits)
            .rev()
            .map(|bit| (gray_int >> bit) & 1 == 1)
            .collect()
    }

    /// Decode a Gray-coded `&[bool]` (MSB first) back to an unsigned integer.
    ///
    /// Uses a running XOR: `binary[i] = binary[i-1] ^ gray[i]`.
    /// Bits beyond `n_bits` are silently ignored.
    pub fn decode(&self, gray: &[bool]) -> u64 {
        let len = gray.len().min(self.n_bits);
        if len == 0 {
            return 0;
        }
        let mut result = 0u64;
        let mut prev_bit = false;
        for &bit in gray.iter().take(len) {
            let cur = bit ^ prev_bit;
            prev_bit = cur;
            result = (result << 1) | (cur as u64);
        }
        // If the supplied slice is shorter than n_bits, left-align the result.
        if len < self.n_bits {
            result <<= self.n_bits - len;
        }
        result
    }

    /// Decode a Gray-coded `&[bool]` and scale the resulting integer into `[lb, ub]`.
    ///
    /// Maps the integer range `[0, 2^n_bits - 1]` linearly onto `[lb, ub]`.
    ///
    /// # Errors
    /// Returns `EvolError::InvalidParameter` if `lb >= ub`.
    pub fn to_float(&self, gray: &[bool], lb: f64, ub: f64) -> EvolResult<f64> {
        if lb >= ub {
            return Err(EvolError::InvalidParameter(format!(
                "GrayEncoder::to_float: lb ({lb}) must be < ub ({ub})"
            )));
        }
        let int_val = self.decode(gray);
        let fraction = int_val as f64 / self.max_value as f64;
        Ok(lb + fraction * (ub - lb))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Permutation crossover operators
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that a permutation is a valid permutation of `0..n`.
fn validate_permutation(perm: &[usize], label: &str) -> EvolResult<()> {
    let n = perm.len();
    if n == 0 {
        return Err(EvolError::EmptyGenome);
    }
    let mut seen = vec![false; n];
    for &v in perm {
        if v >= n {
            return Err(EvolError::InvalidParameter(format!(
                "{label}: value {v} out of range for permutation of length {n}"
            )));
        }
        if seen[v] {
            return Err(EvolError::InvalidParameter(format!(
                "{label}: duplicate value {v} in permutation"
            )));
        }
        seen[v] = true;
    }
    Ok(())
}

/// Choose a random interval `[l, r)` with `l < r` inside a permutation of length `n`.
fn random_interval(n: usize, rng: &mut LcgRng) -> (usize, usize) {
    // Guarantee l < r by construction.
    let l = rng.next_usize(n - 1); // [0, n-2]
    let width = 1 + rng.next_usize(n - l - 1 + 1); // [1, n-l]
    let r = (l + width).min(n);
    (l, r)
}

/// Partially Mapped Crossover (PMX) for permutation genomes.
///
/// 1. Pick a random segment `[l, r)`.
/// 2. Copy that segment from `parent_a` into the child.
/// 3. For each position outside `[l, r)` in `parent_b`, fill the child position with
///    `parent_b`'s value if it hasn't already been placed; otherwise follow the mapping
///    chain until a free value is found.
///
/// # Errors
/// Returns `EvolError::EmptyGenome` or `EvolError::InvalidParameter` for invalid inputs,
/// or `EvolError::DimensionMismatch` if the parents differ in length.
pub fn pmx_crossover(
    parent_a: &[usize],
    parent_b: &[usize],
    rng: &mut LcgRng,
) -> EvolResult<Vec<usize>> {
    validate_permutation(parent_a, "pmx_crossover parent_a")?;
    let n = parent_a.len();
    if parent_b.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: parent_b.len(),
        });
    }
    validate_permutation(parent_b, "pmx_crossover parent_b")?;

    if n == 1 {
        return Ok(vec![0]);
    }

    let (l, r) = random_interval(n, rng);

    // Mark which values are fixed into the copied segment from parent_a.
    let mut in_segment = vec![false; n];
    for i in l..r {
        in_segment[parent_a[i]] = true;
    }

    // Build the PMX mapping: seg_a_to_b[v] = parent_b[k] where parent_a[k] = v, k ∈ [l,r).
    // Used to follow mapping chains when parent_b[i] collides with the copied segment.
    let mut seg_a_to_b = vec![usize::MAX; n];
    for i in l..r {
        seg_a_to_b[parent_a[i]] = parent_b[i];
    }

    // Sentinel: usize::MAX means "not yet placed".
    let mut child = vec![usize::MAX; n];

    // Step 1: Copy segment from parent_a.
    child[l..r].copy_from_slice(&parent_a[l..r]);

    // Step 2: Fill positions outside [l, r) with parent_b values, following mapping chains.
    // For each position i outside the segment, start with parent_b[i]:
    //   - If it is not in the A-segment, place it directly.
    //   - Otherwise, follow the chain through seg_a_to_b until a free value is found.
    for i in (0..l).chain(r..n) {
        let mut val = parent_b[i];
        let mut steps = 0usize;
        while in_segment[val] {
            // val is already placed in the segment; follow the PMX mapping chain.
            val = seg_a_to_b[val];
            steps += 1;
            if steps > n {
                break; // guard against unexpected cycles (should not happen for valid perms)
            }
        }
        child[i] = val;
    }

    Ok(child)
}

/// Order Crossover (OX) for permutation genomes.
///
/// 1. Pick a random segment `[l, r)`.
/// 2. Copy that segment from `parent_a` into the child.
/// 3. Fill the remaining positions in the order they appear in `parent_b`, skipping
///    values already placed.
///
/// # Errors
/// Returns `EvolError::EmptyGenome` or `EvolError::DimensionMismatch` for invalid inputs.
pub fn ox_crossover(
    parent_a: &[usize],
    parent_b: &[usize],
    rng: &mut LcgRng,
) -> EvolResult<Vec<usize>> {
    validate_permutation(parent_a, "ox_crossover parent_a")?;
    let n = parent_a.len();
    if parent_b.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: parent_b.len(),
        });
    }
    validate_permutation(parent_b, "ox_crossover parent_b")?;

    if n == 1 {
        return Ok(vec![0]);
    }

    let (l, r) = random_interval(n, rng);
    let mut child = vec![usize::MAX; n];
    let mut in_segment = vec![false; n];

    // Copy segment from parent_a.
    for i in l..r {
        child[i] = parent_a[i];
        in_segment[parent_a[i]] = true;
    }

    // Collect remaining values in parent_b order.
    let remaining: Vec<usize> = parent_b
        .iter()
        .copied()
        .filter(|&v| !in_segment[v])
        .collect();

    // Fill positions outside [l, r) with remaining values.
    let fill_positions: Vec<usize> = (r..n).chain(0..l).collect();
    for (pos, val) in fill_positions.into_iter().zip(remaining) {
        child[pos] = val;
    }

    Ok(child)
}

/// Cycle Crossover (CX) for permutation genomes.
///
/// Identifies cycles between the two parents by position-to-position matching:
/// a cycle begins at position `i`, then follows `parent_b[i]` to find its position
/// in `parent_a`, and so on until the cycle closes. Odd-numbered cycles are copied
/// from `parent_a`; even-numbered from `parent_b`.
///
/// # Errors
/// Returns `EvolError::EmptyGenome` or `EvolError::DimensionMismatch` for invalid inputs.
pub fn cx_crossover(parent_a: &[usize], parent_b: &[usize]) -> EvolResult<Vec<usize>> {
    validate_permutation(parent_a, "cx_crossover parent_a")?;
    let n = parent_a.len();
    if parent_b.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: parent_b.len(),
        });
    }
    validate_permutation(parent_b, "cx_crossover parent_b")?;

    if n == 1 {
        return Ok(vec![0]);
    }

    // Build position lookup: pos_in_a[val] = index where val appears in parent_a.
    let mut pos_in_a = vec![0usize; n];
    for (i, &v) in parent_a.iter().enumerate() {
        pos_in_a[v] = i;
    }

    let mut assigned = vec![false; n];
    let mut child = vec![0usize; n];
    let mut cycle_idx = 0usize;

    for start in 0..n {
        if assigned[start] {
            continue;
        }
        // Trace cycle starting at position `start`.
        let mut cycle_positions = Vec::new();
        let mut pos = start;
        loop {
            cycle_positions.push(pos);
            assigned[pos] = true;
            // Follow: from current position, find where parent_b[pos] lives in parent_a.
            let next_pos = pos_in_a[parent_b[pos]];
            if next_pos == start {
                break;
            }
            pos = next_pos;
        }
        // Odd cycles (0-indexed) come from parent_a; even cycles from parent_b.
        for &cp in &cycle_positions {
            child[cp] = if cycle_idx.is_multiple_of(2) {
                parent_a[cp]
            } else {
                parent_b[cp]
            };
        }
        cycle_idx += 1;
    }

    Ok(child)
}

// ─────────────────────────────────────────────────────────────────────────────
// Permutation mutation operators
// ─────────────────────────────────────────────────────────────────────────────

/// Inversion mutation (also known as reversal mutation).
///
/// Picks a random segment `[l, r)` and reverses it in-place. This is equivalent to
/// one "2-opt move" without the distance-based acceptance criterion.
///
/// # Errors
/// Returns `EvolError::EmptyGenome` if `perm` is empty.
pub fn inversion_mutation(perm: &mut [usize], rng: &mut LcgRng) -> EvolResult<()> {
    let n = perm.len();
    if n == 0 {
        return Err(EvolError::EmptyGenome);
    }
    if n == 1 {
        return Ok(());
    }
    let (l, r) = random_interval(n, rng);
    perm[l..r].reverse();
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-opt local search
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the total tour cost for a given ordering using the provided cost matrix.
fn tour_cost(tour: &[usize], cost_matrix: &[Vec<f64>]) -> f64 {
    let n = tour.len();
    if n == 0 {
        return 0.0;
    }
    (0..n)
        .map(|i| cost_matrix[tour[i]][tour[(i + 1) % n]])
        .sum()
}

/// Single-pass 2-opt improvement for TSP-style tours.
///
/// Tries all pairs `(i, j)` with `i < j - 1`; accepts the **first** swap that strictly
/// decreases the tour cost (first-improvement strategy). The reversed segment is
/// `tour[i+1..=j]`. Returns the improved tour (or the original if no improvement exists).
///
/// Complexity: O(n²) per call in the worst case.
///
/// # Errors
/// Returns `EvolError::EmptyGenome` if `tour` is empty.
/// Returns `EvolError::DimensionMismatch` if the cost matrix dimensions do not match
/// the number of cities.
pub fn two_opt_improve(tour: &[usize], cost_matrix: &[Vec<f64>]) -> EvolResult<Vec<usize>> {
    let n = tour.len();
    if n == 0 {
        return Err(EvolError::EmptyGenome);
    }
    // Validate cost matrix dimensions.
    if cost_matrix.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: cost_matrix.len(),
        });
    }
    for (row_idx, row) in cost_matrix.iter().enumerate() {
        if row.len() != n {
            return Err(EvolError::DimensionMismatch {
                expected: n,
                got: row.len(),
            });
        }
        // Validate that tour indices are in-range.
        if tour[row_idx] >= n {
            return Err(EvolError::InvalidParameter(format!(
                "two_opt_improve: tour index {} >= n={n}",
                tour[row_idx]
            )));
        }
    }

    let current_cost = tour_cost(tour, cost_matrix);
    let mut best_tour = tour.to_vec();
    let mut best_cost = current_cost;

    'outer: for i in 0..n - 1 {
        for j in (i + 2)..n {
            // Skip the wrap-around edge when i==0 and j==n-1 (same edge).
            if i == 0 && j == n - 1 {
                continue;
            }
            // Compute delta: remove edges (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%n]),
            // add (tour[i], tour[j]) and (tour[i+1], tour[(j+1)%n]).
            let a = best_tour[i];
            let b = best_tour[i + 1];
            let c = best_tour[j];
            let d = best_tour[(j + 1) % n];

            let old_cost = cost_matrix[a][b] + cost_matrix[c][d];
            let new_cost = cost_matrix[a][c] + cost_matrix[b][d];

            if new_cost < old_cost - 1e-14 {
                // Accept: reverse segment [i+1..=j].
                best_tour[i + 1..=j].reverse();
                best_cost = best_cost - old_cost + new_cost;
                let _ = best_cost; // suppress unused warning
                break 'outer;
            }
        }
    }

    Ok(best_tour)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn is_valid_permutation(perm: &[usize]) -> bool {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &v in perm {
            if v >= n || seen[v] {
                return false;
            }
            seen[v] = true;
        }
        true
    }

    // ── GrayEncoder tests ──────────────────────────────────────────────────

    #[test]
    fn test_gray_encoder_new_zero_bits_errors() {
        assert!(GrayEncoder::new(0).is_err());
    }

    #[test]
    fn test_gray_encoder_new_too_many_bits_errors() {
        assert!(GrayEncoder::new(64).is_err());
    }

    #[test]
    fn test_gray_encoder_encode_decode_roundtrip_8bit() {
        let enc = GrayEncoder::new(8).expect("new should succeed");
        for v in 0u64..=255 {
            let gray = enc.encode(v);
            assert_eq!(gray.len(), 8);
            assert_eq!(enc.decode(&gray), v, "round-trip failed for {v}");
        }
    }

    #[test]
    fn test_gray_encoder_adjacent_values_differ_by_one_bit() {
        let enc = GrayEncoder::new(8).expect("new should succeed");
        for v in 0u64..255 {
            let g1 = enc.encode(v);
            let g2 = enc.encode(v + 1);
            let hamming: usize = g1
                .iter()
                .zip(g2.iter())
                .map(|(a, b)| (a != b) as usize)
                .sum();
            assert_eq!(hamming, 1, "Gray code property violated at {v}->{}", v + 1);
        }
    }

    #[test]
    fn test_gray_encoder_encode_zero_is_all_false() {
        let enc = GrayEncoder::new(4).expect("new should succeed");
        let gray = enc.encode(0);
        assert!(gray.iter().all(|&b| !b));
    }

    #[test]
    fn test_gray_encoder_to_float_lb_ub_range() {
        let enc = GrayEncoder::new(8).expect("new should succeed");
        let gray_min = enc.encode(0);
        let gray_max = enc.encode(255);
        let f_min = enc
            .to_float(&gray_min, -1.0, 1.0)
            .expect("to_float should succeed");
        let f_max = enc
            .to_float(&gray_max, -1.0, 1.0)
            .expect("to_float should succeed");
        assert!((f_min - (-1.0)).abs() < 1e-9);
        assert!((f_max - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_gray_encoder_to_float_invalid_bounds_errors() {
        let enc = GrayEncoder::new(4).expect("new should succeed");
        let gray = enc.encode(5);
        assert!(enc.to_float(&gray, 1.0, 1.0).is_err());
        assert!(enc.to_float(&gray, 2.0, 1.0).is_err());
    }

    #[test]
    fn test_gray_encoder_mask_overflow() {
        // Encoding a value larger than max should mask to n_bits.
        let enc = GrayEncoder::new(4).expect("new should succeed");
        // 16 masked to 4 bits == 0.
        let gray = enc.encode(16);
        assert_eq!(enc.decode(&gray), 0);
    }

    // ── PMX crossover tests ────────────────────────────────────────────────

    #[test]
    fn test_pmx_crossover_produces_valid_permutation() {
        let mut rng = LcgRng::new(42);
        let pa: Vec<usize> = (0..8).collect();
        let pb: Vec<usize> = vec![3, 7, 2, 1, 4, 0, 5, 6];
        for _ in 0..20 {
            let child = pmx_crossover(&pa, &pb, &mut rng).expect("pmx_crossover should succeed");
            assert!(is_valid_permutation(&child), "PMX child invalid: {child:?}");
        }
    }

    #[test]
    fn test_pmx_crossover_dimension_mismatch_errors() {
        let mut rng = LcgRng::new(1);
        let pa = vec![0, 1, 2];
        let pb = vec![0, 1];
        assert!(pmx_crossover(&pa, &pb, &mut rng).is_err());
    }

    #[test]
    fn test_pmx_crossover_empty_errors() {
        let mut rng = LcgRng::new(1);
        let empty: Vec<usize> = vec![];
        assert!(pmx_crossover(&empty, &empty, &mut rng).is_err());
    }

    // ── OX crossover tests ─────────────────────────────────────────────────

    #[test]
    fn test_ox_crossover_produces_valid_permutation() {
        let mut rng = LcgRng::new(99);
        let pa: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let pb: Vec<usize> = vec![7, 6, 5, 4, 3, 2, 1, 0];
        for _ in 0..20 {
            let child = ox_crossover(&pa, &pb, &mut rng).expect("ox_crossover should succeed");
            assert!(is_valid_permutation(&child), "OX child invalid: {child:?}");
        }
    }

    #[test]
    fn test_ox_crossover_segment_from_parent_a() {
        // With a fixed rng, the segment [l,r) from parent_a must appear verbatim in child.
        let mut rng = LcgRng::new(7);
        let pa: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6];
        let pb: Vec<usize> = vec![6, 5, 4, 3, 2, 1, 0];
        let child = ox_crossover(&pa, &pb, &mut rng).expect("ox_crossover should succeed");
        assert!(is_valid_permutation(&child));
    }

    #[test]
    fn test_ox_crossover_dimension_mismatch_errors() {
        let mut rng = LcgRng::new(1);
        let pa = vec![0, 1, 2];
        let pb = vec![0, 1, 2, 3];
        assert!(ox_crossover(&pa, &pb, &mut rng).is_err());
    }

    // ── CX crossover tests ─────────────────────────────────────────────────

    #[test]
    fn test_cx_crossover_produces_valid_permutation() {
        let pa: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let pb: Vec<usize> = vec![3, 7, 2, 1, 4, 0, 5, 6];
        let child = cx_crossover(&pa, &pb).expect("cx_crossover should succeed");
        assert!(is_valid_permutation(&child), "CX child invalid: {child:?}");
    }

    #[test]
    fn test_cx_crossover_values_from_parents_only() {
        // Each gene in child must come from parent_a or parent_b at the same position.
        let pa: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let pb: Vec<usize> = vec![4, 5, 6, 7, 0, 1, 2, 3];
        let child = cx_crossover(&pa, &pb).expect("cx_crossover should succeed");
        for (i, &v) in child.iter().enumerate() {
            assert!(v == pa[i] || v == pb[i], "pos {i}: {v} not in parents");
        }
    }

    #[test]
    fn test_cx_crossover_identity_parents_yields_identity() {
        let pa: Vec<usize> = vec![0, 1, 2, 3, 4];
        let pb = pa.clone();
        let child = cx_crossover(&pa, &pb).expect("cx_crossover should succeed");
        assert_eq!(child, pa);
    }

    #[test]
    fn test_cx_crossover_dimension_mismatch_errors() {
        let pa = vec![0, 1, 2];
        let pb = vec![0, 1, 2, 3];
        assert!(cx_crossover(&pa, &pb).is_err());
    }

    #[test]
    fn test_cx_crossover_single_element() {
        let pa = vec![0];
        let pb = vec![0];
        let child = cx_crossover(&pa, &pb).expect("cx_crossover should succeed");
        assert_eq!(child, vec![0]);
    }

    // ── Inversion mutation tests ───────────────────────────────────────────

    #[test]
    fn test_inversion_mutation_preserves_permutation() {
        let mut rng = LcgRng::new(55);
        let mut perm: Vec<usize> = (0..10).collect();
        for _ in 0..30 {
            inversion_mutation(&mut perm, &mut rng).expect("inversion_mutation should succeed");
            assert!(is_valid_permutation(&perm), "inversion broke permutation");
        }
    }

    #[test]
    fn test_inversion_mutation_empty_errors() {
        let mut rng = LcgRng::new(1);
        let mut empty: Vec<usize> = vec![];
        assert!(inversion_mutation(&mut empty, &mut rng).is_err());
    }

    #[test]
    fn test_inversion_mutation_single_element_is_noop() {
        let mut rng = LcgRng::new(1);
        let mut perm = vec![0usize];
        inversion_mutation(&mut perm, &mut rng).expect("inversion_mutation should succeed");
        assert_eq!(perm, vec![0]);
    }

    // ── 2-opt improvement tests ───────────────────────────────────────────

    #[test]
    fn test_two_opt_improve_never_increases_cost() {
        // A simple 4-city symmetric tour where one swap helps.
        // Positions: 0(0,0), 1(1,0), 2(1,1), 3(0,1). Sub-optimal tour: 0→2→1→3→0.
        let cost: Vec<Vec<f64>> = vec![
            vec![0.0, 1.0, 1.414, 1.0],
            vec![1.0, 0.0, 1.0, 1.414],
            vec![1.414, 1.0, 0.0, 1.0],
            vec![1.0, 1.414, 1.0, 0.0],
        ];
        let tour = vec![0, 2, 1, 3]; // sub-optimal
        let original_cost = tour_cost(&tour, &cost);
        let improved = two_opt_improve(&tour, &cost).expect("two_opt_improve should succeed");
        let improved_cost = tour_cost(&improved, &cost);
        assert!(
            improved_cost <= original_cost + 1e-10,
            "2-opt increased cost: {original_cost} → {improved_cost}"
        );
        assert!(is_valid_permutation(&improved));
    }

    #[test]
    fn test_two_opt_improve_already_optimal_unchanged() {
        // Identity tour on 4 cities with zero-diagonal cost.
        let cost: Vec<Vec<f64>> = vec![
            vec![0.0, 1.0, 2.0, 1.5],
            vec![1.0, 0.0, 1.0, 2.0],
            vec![2.0, 1.0, 0.0, 1.0],
            vec![1.5, 2.0, 1.0, 0.0],
        ];
        let tour = vec![0, 1, 2, 3]; // already optimal for this metric
        let original_cost = tour_cost(&tour, &cost);
        let improved = two_opt_improve(&tour, &cost).expect("two_opt_improve should succeed");
        let improved_cost = tour_cost(&improved, &cost);
        assert!(improved_cost <= original_cost + 1e-10);
    }

    #[test]
    fn test_two_opt_improve_empty_tour_errors() {
        let tour: Vec<usize> = vec![];
        let cost: Vec<Vec<f64>> = vec![];
        assert!(two_opt_improve(&tour, &cost).is_err());
    }

    #[test]
    fn test_two_opt_improve_dimension_mismatch_errors() {
        let tour = vec![0, 1, 2];
        let cost: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![1.0, 0.0]]; // 2×2 ≠ 3×3
        assert!(two_opt_improve(&tour, &cost).is_err());
    }

    #[test]
    fn test_two_opt_improve_result_is_valid_permutation() {
        let cost: Vec<Vec<f64>> = vec![
            vec![0.0, 2.0, 9.0, 10.0],
            vec![2.0, 0.0, 6.0, 4.0],
            vec![9.0, 6.0, 0.0, 8.0],
            vec![10.0, 4.0, 8.0, 0.0],
        ];
        let tour = vec![0, 3, 1, 2];
        let improved = two_opt_improve(&tour, &cost).expect("two_opt_improve should succeed");
        assert!(is_valid_permutation(&improved));
    }
}
