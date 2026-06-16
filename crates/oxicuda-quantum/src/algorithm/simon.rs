//! Simon's algorithm.
//!
//! Given a two-to-one (or one-to-one) Boolean function `f : {0,1}^n → {0,1}^n`
//! with the promise `f(x) = f(y) ⇔ y = x ⊕ s` for some hidden period
//! `s ∈ {0,1}^n`, Simon's algorithm recovers `s` with `O(n)` quantum queries —
//! an *exponential* speed-up over the `Θ(2^{n/2})` classical lower bound. It is
//! the historical precursor to Shor's algorithm: a single run of the quantum
//! circuit returns a uniformly random bit-string `y` satisfying `y · s = 0
//! (mod 2)`, and after collecting `n − 1` linearly independent such constraints
//! the hidden string is fixed (up to the trivial all-zero solution when
//! `s = 0`).
//!
//! ## Circuit (on `2n` qubits)
//!
//! The first `n` qubits form the *query* register, the second `n` the *output*
//! register (this crate is **little-endian**, so query bit `i` ↦ qubit `i` and
//! output bit `j` ↦ qubit `n + j`):
//!
//! 1. Prepare `H^{⊗n}` on the query register.
//! 2. Apply the oracle `U_f|x⟩|0⟩ = |x⟩|f(x)⟩`.
//! 3. Apply `H^{⊗n}` on the query register again.
//! 4. Measure the query register — the outcome `y` is orthogonal to `s`.
//!
//! ## Classical post-processing
//!
//! The sampled constraints are assembled into a binary matrix and reduced to
//! row-echelon form over `GF(2)`; the unique non-trivial vector in the kernel is
//! the period `s`. [`recover_period_from_constraints`] exposes this solver
//! independently of the quantum sampler.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::hadamard::gate_h;
use crate::handle::LcgRng;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Outcome of [`simon`].
#[derive(Debug, Clone)]
pub struct SimonResult {
    /// The recovered hidden period `s` (little-endian: bit `i` ↦ query qubit `i`).
    ///
    /// Equals `0` when the oracle is one-to-one (the trivial period).
    pub period: usize,
    /// The linearly independent constraint vectors `y` sampled from the circuit
    /// (each satisfies `y · s = 0 (mod 2)`).
    pub constraints: Vec<usize>,
    /// Number of circuit executions (shots) consumed to gather the constraints.
    pub shots_used: usize,
}

/// Run one Simon circuit and return a single sampled constraint `y`.
///
/// The output register is *not* measured explicitly; instead the query register
/// is sampled after the final Hadamard layer. Because `f` is realised exactly on
/// the amplitude array, the marginal distribution of the query register matches
/// the textbook `y · s = 0` support.
fn sample_constraint<F>(oracle: F, n: usize, rng: &mut LcgRng) -> QuantumResult<usize>
where
    F: Fn(usize) -> usize,
{
    let total_qubits = 2 * n;
    let mut sv = StateVector::new_zero_state(total_qubits)?;

    // Step 1: Hadamard the query register (qubits 0..n).
    for q in 0..n {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    // Step 2: oracle U_f|x⟩|0⟩ = |x⟩|f(x)⟩. The input output register is |0⟩, so
    // amplitude on basis index `x` (query) maps to `x | (f(x) << n)`.
    let dim = 1usize << total_qubits;
    let mut new_amps = vec![num_complex::Complex::<f32>::new(0.0, 0.0); dim];
    let query_mask = (1usize << n) - 1;
    let out_mask = ((1usize << n) - 1) << n;
    for (idx, a) in sv.amps.iter().enumerate() {
        if a.norm_sqr() == 0.0 {
            continue;
        }
        let x = idx & query_mask;
        let y_in = (idx & out_mask) >> n;
        let fx = oracle(x) & ((1usize << n) - 1);
        let y_out = y_in ^ fx;
        let target = x | (y_out << n);
        new_amps[target] += *a;
    }
    sv.amps = new_amps;

    // Step 3: re-apply Hadamards to the query register.
    for q in 0..n {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    // Step 4: sample the query register bit-by-bit, collapsing as we go.
    let mut y = 0usize;
    for q in 0..n {
        let (bit, collapsed) = sv.sample_measure(q, rng)?;
        sv = collapsed;
        if bit {
            y |= 1 << q;
        }
    }
    Ok(y)
}

/// Insert constraint `y` into a `GF(2)` row-echelon basis, returning `true` when
/// it was linearly independent of the existing rows (and hence added).
fn gf2_insert(basis: &mut Vec<usize>, mut y: usize) -> bool {
    if y == 0 {
        return false;
    }
    for &row in basis.iter() {
        // Reduce `y` by the pivot (lowest set bit) of `row`.
        let pivot = row & row.wrapping_neg();
        if (y & pivot) != 0 {
            y ^= row;
        }
        if y == 0 {
            return false;
        }
    }
    basis.push(y);
    true
}

/// Recover the hidden period `s` from a set of constraint vectors `y` (each
/// satisfying `y · s = 0 (mod 2)`) over `n` bits.
///
/// Solves the homogeneous system `Y s = 0` over `GF(2)` by Gaussian elimination
/// and returns the unique non-trivial kernel vector. If the constraints do not
/// pin down a single non-zero solution (under-determined), the all-zero period
/// `0` is returned, signalling either a one-to-one oracle or insufficient rank.
///
/// # Errors
/// * [`QuantumError::InvalidQubitCount`] if `n` is `0` or `> 30`.
pub fn recover_period_from_constraints(constraints: &[usize], n: usize) -> QuantumResult<usize> {
    if n == 0 || n > 30 {
        return Err(QuantumError::InvalidQubitCount { n });
    }
    // Build a reduced row-echelon basis over GF(2).
    let mut basis: Vec<usize> = Vec::new();
    for &c in constraints {
        let masked = c & ((1usize << n) - 1);
        gf2_insert(&mut basis, masked);
    }
    let rank = basis.len();
    // The solution space has dimension n - rank. A unique non-trivial period
    // exists exactly when rank == n - 1.
    if rank != n - 1 {
        return Ok(0);
    }

    // Full reduce to RREF so each pivot column is isolated.
    let mut rref = basis.clone();
    for i in 0..rref.len() {
        let pivot_i = rref[i] & rref[i].wrapping_neg();
        for j in 0..rref.len() {
            if i != j && (rref[j] & pivot_i) != 0 {
                rref[j] ^= rref[i];
            }
        }
    }
    let pivot_cols: Vec<usize> = rref
        .iter()
        .map(|&r| (r & r.wrapping_neg()).trailing_zeros() as usize)
        .collect();

    // The single free column is the one not used as a pivot.
    let mut free_col = None;
    for col in 0..n {
        if !pivot_cols.contains(&col) {
            free_col = Some(col);
            break;
        }
    }
    let free_col = match free_col {
        Some(c) => c,
        None => return Ok(0),
    };

    // Set the free variable to 1; each pivot variable equals the coefficient of
    // the free column in its row.
    let mut s = 1usize << free_col;
    for (row_idx, &row) in rref.iter().enumerate() {
        if (row >> free_col) & 1 == 1 {
            s |= 1 << pivot_cols[row_idx];
        }
    }
    Ok(s & ((1usize << n) - 1))
}

/// Run Simon's algorithm to recover the hidden period of a two-to-one oracle.
///
/// `oracle(x)` returns `f(x)` for query index `x ∈ [0, 2^n)`, where `f` obeys
/// the Simon promise `f(x) = f(y) ⇔ y = x ⊕ s`. The routine repeatedly runs the
/// quantum circuit (seeded by `rng`) until `n − 1` linearly independent
/// constraints have been collected (or `max_shots` is exhausted), then solves
/// for `s` classically.
///
/// # Errors
/// * [`QuantumError::InvalidQubitCount`] if `n` is `0` or `> 15` (the circuit
///   uses `2n` qubits; `2n ≤ 30`).
/// * [`QuantumError::MeasurementFailed`] is propagated from the sampler.
pub fn simon<F>(
    oracle: F,
    n: usize,
    max_shots: usize,
    rng: &mut LcgRng,
) -> QuantumResult<SimonResult>
where
    F: Fn(usize) -> usize,
{
    if n == 0 || n > 15 {
        return Err(QuantumError::InvalidQubitCount { n });
    }
    let mut basis: Vec<usize> = Vec::new();
    let mut shots_used = 0usize;
    while basis.len() < n - 1 && shots_used < max_shots {
        let y = sample_constraint(&oracle, n, rng)?;
        shots_used += 1;
        gf2_insert(&mut basis, y);
    }
    let period = recover_period_from_constraints(&basis, n)?;
    Ok(SimonResult {
        period,
        constraints: basis,
        shots_used,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a two-to-one oracle with period `s` over `n` bits.
    ///
    /// `f(x) = min(x, x ⊕ s)` is constant exactly on each coset `{x, x ⊕ s}`,
    /// so it satisfies the Simon promise for any non-zero `s`.
    fn two_to_one(s: usize) -> impl Fn(usize) -> usize {
        move |x: usize| x.min(x ^ s)
    }

    #[test]
    fn gf2_insert_rejects_zero() {
        let mut basis = Vec::new();
        assert!(!gf2_insert(&mut basis, 0));
        assert!(basis.is_empty());
    }

    #[test]
    fn gf2_insert_detects_dependence() {
        let mut basis = Vec::new();
        assert!(gf2_insert(&mut basis, 0b011));
        assert!(gf2_insert(&mut basis, 0b101));
        // 0b110 = 0b011 ⊕ 0b101 is dependent.
        assert!(!gf2_insert(&mut basis, 0b110));
        assert_eq!(basis.len(), 2);
    }

    #[test]
    fn recover_period_simple_2bit() {
        // s = 0b01; constraints orthogonal to it: y · s = 0 ⇒ y0 = 0 ⇒ y = 0b10.
        let s = recover_period_from_constraints(&[0b10], 2)
            .expect("single valid 2-bit constraint for n=2");
        assert_eq!(s, 0b01);
    }

    #[test]
    fn recover_period_3bit() {
        // s = 0b101. Two independent constraints orthogonal to s.
        // y·s = y0 ⊕ y2 = 0. Pick y = 0b010 (y0=0,y2=0) and y = 0b101 (1⊕1=0).
        let s = recover_period_from_constraints(&[0b010, 0b101], 3)
            .expect("two valid 3-bit constraints for n=3");
        assert_eq!(s, 0b101);
    }

    #[test]
    fn recover_period_underdetermined_returns_zero() {
        // Only one constraint for n=3 ⇒ rank 1 ≠ n-1=2 ⇒ trivial.
        let s = recover_period_from_constraints(&[0b010], 3)
            .expect("single constraint for n=3 is valid, returns 0 for underdetermined");
        assert_eq!(s, 0);
    }

    #[test]
    fn recover_period_invalid_n() {
        assert!(recover_period_from_constraints(&[1], 0).is_err());
    }

    #[test]
    fn sample_constraint_is_orthogonal() {
        let s = 0b11usize;
        let mut rng = LcgRng::new(7);
        for _ in 0..20 {
            let y = sample_constraint(two_to_one(s), 2, &mut rng)
                .expect("2-qubit Simon circuit with valid two-to-one oracle");
            assert_eq!((y & s).count_ones() & 1, 0, "y={y:b} not orthogonal to s");
        }
    }

    #[test]
    fn simon_recovers_period_2bit() {
        let s = 0b01usize;
        let mut rng = LcgRng::new(123);
        let res = simon(two_to_one(s), 2, 200, &mut rng)
            .expect("simon with valid 2-qubit oracle and 200 shots");
        assert_eq!(res.period, s, "constraints={:?}", res.constraints);
    }

    #[test]
    fn simon_recovers_period_3bit() {
        let s = 0b110usize;
        let mut rng = LcgRng::new(456);
        let res = simon(two_to_one(s), 3, 500, &mut rng)
            .expect("simon with valid 3-qubit oracle and 500 shots");
        assert_eq!(res.period, s, "constraints={:?}", res.constraints);
    }

    #[test]
    fn simon_constraints_all_orthogonal() {
        let s = 0b101usize;
        let mut rng = LcgRng::new(999);
        let res = simon(two_to_one(s), 3, 500, &mut rng)
            .expect("simon with valid 3-qubit oracle and 500 shots");
        for &y in &res.constraints {
            assert_eq!((y & s).count_ones() & 1, 0, "y={y:b} not orthogonal");
        }
    }

    #[test]
    fn simon_collects_n_minus_one_constraints() {
        let s = 0b110usize;
        let mut rng = LcgRng::new(2024);
        let res = simon(two_to_one(s), 3, 500, &mut rng)
            .expect("simon with valid 3-qubit oracle and 500 shots");
        assert_eq!(
            res.constraints.len(),
            2,
            "should reach n-1 independent rows"
        );
        assert!(res.shots_used >= 2);
    }

    #[test]
    fn simon_invalid_qubit_count() {
        let mut rng = LcgRng::new(1);
        assert!(simon(|x| x, 0, 10, &mut rng).is_err());
        assert!(simon(|x| x, 16, 10, &mut rng).is_err());
    }

    #[test]
    fn simon_deterministic_with_same_seed() {
        let s = 0b11usize;
        let mut rng_a = LcgRng::new(55);
        let mut rng_b = LcgRng::new(55);
        let a = simon(two_to_one(s), 2, 200, &mut rng_a)
            .expect("simon with valid 2-qubit oracle, rng_a");
        let b = simon(two_to_one(s), 2, 200, &mut rng_b)
            .expect("simon with valid 2-qubit oracle, rng_b");
        assert_eq!(a.period, b.period);
        assert_eq!(a.constraints, b.constraints);
    }
}
