//! Grover's unstructured search algorithm.
//!
//! Given a black-box *oracle* that marks a subset of computational-basis states
//! `|w⟩` (the "solutions") out of `N = 2^n` items, Grover's algorithm finds a
//! marked item with high probability using only `Θ(√(N/M))` oracle queries
//! (`M` = number of marked items), a quadratic speed-up over the classical
//! `Θ(N/M)`.
//!
//! The circuit prepares the uniform superposition `H^{⊗n}|0⟩`, then repeats the
//! *Grover iterate* `G = D · O`:
//!
//! * **Oracle `O`** flips the sign of every marked basis amplitude:
//!   `O|x⟩ = (-1)^{f(x)}|x⟩` where `f(x)=1` iff `x` is marked.
//! * **Diffusion `D`** (inversion about the mean) reflects every amplitude about
//!   the average amplitude: `D = 2|s⟩⟨s| - I` with `|s⟩ = H^{⊗n}|0⟩`. Acting on
//!   an amplitude vector this is `a_x ↦ 2·⟨a⟩ - a_x`.
//!
//! The optimal number of iterations is `r = round(π/4 · √(N/M) - 1/2)`, which
//! rotates the state vector to nearly coincide with the equal superposition of
//! marked items. Reading out the register then returns a marked index with
//! probability `≈ 1`.
//!
//! This implementation is **state-vector exact**: the oracle and diffusion are
//! applied directly to the amplitude array (rather than synthesised from a
//! multi-controlled-Z gate net), which keeps the routine fast and numerically
//! clean while reproducing the textbook amplitude trajectory precisely.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::hadamard::gate_h;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Outcome of [`grover_search`].
#[derive(Debug, Clone)]
pub struct GroverResult {
    /// The basis index read out by the (deterministic) argmax measurement.
    pub measured_index: usize,
    /// Probability mass concentrated on `measured_index` after the iterations.
    pub probability: f32,
    /// Number of Grover iterations actually performed.
    pub iterations: usize,
}

/// Compute the optimal Grover iteration count for `n_marked` solutions in a
/// space of `dim = 2^n_qubits` items.
///
/// Returns `round(π/4 · √(dim / n_marked) - 1/2)`, clamped to be at least one
/// when at least one item is marked and fewer than half the space is marked.
/// When `n_marked == 0` (no solutions) the count is `0`. When `n_marked`
/// approaches `dim` the analytic formula yields `0`, which is correct: the
/// uniform state already has the marked mass.
#[must_use]
pub fn optimal_iterations(dim: usize, n_marked: usize) -> usize {
    if n_marked == 0 || dim == 0 {
        return 0;
    }
    let ratio = dim as f64 / n_marked as f64;
    // θ defined by sin²θ = M/N; the state needs to rotate to π/2.
    let theta = (n_marked as f64 / dim as f64).sqrt().asin();
    if theta <= 0.0 {
        return 0;
    }
    let raw = (std::f64::consts::FRAC_PI_2 / theta - 1.0) * 0.5;
    let rounded = raw.round();
    if rounded <= 0.0 {
        // For ratio just above 1 (almost everything marked) zero is correct;
        // otherwise guarantee at least one amplification step.
        if ratio >= 2.0 { 1 } else { 0 }
    } else {
        rounded as usize
    }
}

/// Validate the oracle and qubit count for a Grover run.
fn validate(oracle_marks: &[usize], n_qubits: usize) -> QuantumResult<usize> {
    if n_qubits == 0 || n_qubits > 30 {
        return Err(QuantumError::InvalidQubitCount { n: n_qubits });
    }
    let dim = 1usize << n_qubits;
    for &m in oracle_marks {
        if m >= dim {
            return Err(QuantumError::InvalidParameter {
                name: format!("marked index {m} out of range for dim {dim}"),
            });
        }
    }
    Ok(dim)
}

/// Apply the phase oracle: negate the amplitude of every marked basis state.
fn apply_oracle(sv: &mut StateVector, marks: &[bool]) {
    for (a, &is_marked) in sv.amps.iter_mut().zip(marks.iter()) {
        if is_marked {
            *a = -*a;
        }
    }
}

/// Apply the diffusion operator (inversion about the mean) in place.
fn apply_diffusion(sv: &mut StateVector) {
    let dim = sv.amps.len();
    if dim == 0 {
        return;
    }
    let inv_dim = 1.0 / dim as f32;
    let mut mean = num_complex::Complex::<f32>::new(0.0, 0.0);
    for a in &sv.amps {
        mean += *a;
    }
    mean *= inv_dim;
    let two_mean = mean * 2.0;
    for a in &mut sv.amps {
        *a = two_mean - *a;
    }
}

/// Run Grover's search and return the most probable basis index.
///
/// Builds the uniform superposition `H^{⊗n}|0⟩`, applies the optimal number of
/// Grover iterates, then performs a deterministic argmax readout over the full
/// register. `oracle_marks` is the (possibly empty, possibly multi-element) list
/// of marked computational-basis indices.
///
/// When `oracle_marks` is empty no amplification occurs and the routine returns
/// the argmax of the uniform state (index `0`) with probability `1/N`.
///
/// # Errors
/// * [`QuantumError::InvalidQubitCount`] if `n_qubits` is `0` or `> 30`.
/// * [`QuantumError::InvalidParameter`] if any marked index is `>= 2^n_qubits`.
pub fn grover_search(oracle_marks: &[usize], n_qubits: usize) -> QuantumResult<GroverResult> {
    let dim = validate(oracle_marks, n_qubits)?;

    // Build a boolean mark-vector for O(dim) oracle application.
    let mut marks = vec![false; dim];
    for &m in oracle_marks {
        marks[m] = true;
    }
    let n_marked = marks.iter().filter(|&&b| b).count();

    // Prepare uniform superposition |s⟩ = H^{⊗n}|0⟩.
    let mut sv = StateVector::new_zero_state(n_qubits)?;
    for q in 0..n_qubits {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    let iterations = optimal_iterations(dim, n_marked);
    for _ in 0..iterations {
        apply_oracle(&mut sv, &marks);
        apply_diffusion(&mut sv);
    }

    // Deterministic argmax readout (ties resolved by lowest index).
    let mut measured_index = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, a) in sv.amps.iter().enumerate() {
        let p = a.norm_sqr();
        if p > best {
            best = p;
            measured_index = i;
        }
    }

    Ok(GroverResult {
        measured_index,
        probability: best,
        iterations,
    })
}

/// Total probability mass currently residing on the marked items.
///
/// Re-runs the amplification and sums `|a_x|²` over marked `x`; useful for
/// asserting that amplitude has been amplified onto the solution subspace.
///
/// # Errors
/// Same conditions as [`grover_search`].
pub fn marked_probability(oracle_marks: &[usize], n_qubits: usize) -> QuantumResult<f32> {
    let dim = validate(oracle_marks, n_qubits)?;
    let mut marks = vec![false; dim];
    for &m in oracle_marks {
        marks[m] = true;
    }
    let n_marked = marks.iter().filter(|&&b| b).count();

    let mut sv = StateVector::new_zero_state(n_qubits)?;
    for q in 0..n_qubits {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }
    let iterations = optimal_iterations(dim, n_marked);
    for _ in 0..iterations {
        apply_oracle(&mut sv, &marks);
        apply_diffusion(&mut sv);
    }

    let mass = sv
        .amps
        .iter()
        .zip(marks.iter())
        .filter(|&(_, &m)| m)
        .map(|(a, _)| a.norm_sqr())
        .sum::<f32>();
    Ok(mass)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_marked_single_solution() {
        // 3-qubit space, mark index 5; Grover must return it.
        let res = grover_search(&[5], 3).expect("valid 3-qubit grover search for mark 5");
        assert_eq!(res.measured_index, 5, "got {}", res.measured_index);
        assert!(res.probability > 0.9, "prob={}", res.probability);
    }

    #[test]
    fn amplitude_amplified_above_uniform() {
        // After amplification, marked mass must far exceed the uniform baseline 1/N.
        let n = 4;
        let uniform = 1.0 / (1u32 << n) as f32;
        let mass =
            marked_probability(&[9], n).expect("valid marked probability for 4-qubit single mark");
        assert!(
            mass > uniform * 4.0,
            "mass={mass} not amplified over uniform {uniform}"
        );
        assert!(mass > 0.9, "mass={mass}");
    }

    #[test]
    fn optimal_iterations_matches_known_values() {
        // n=2 (N=4), single mark: exactly 1 iteration gives certainty.
        assert_eq!(optimal_iterations(4, 1), 1);
        // n=3 (N=8), single mark: round(π/4·√8 - 1/2) = round(1.72) = 2.
        assert_eq!(optimal_iterations(8, 1), 2);
        // n=4 (N=16), single mark: round(π/4·√16 - 1/2) = round(2.64) = 3.
        assert_eq!(optimal_iterations(16, 1), 3);
    }

    #[test]
    fn multiple_marked_items_found() {
        // Mark two of eight items; the returned index must be one of them and
        // the marked subspace must hold most of the probability.
        let marks = [2usize, 6];
        let res = grover_search(&marks, 3).expect("valid 3-qubit grover search for marks [2, 6]");
        assert!(
            marks.contains(&res.measured_index),
            "got {}",
            res.measured_index
        );
        let mass = marked_probability(&marks, 3)
            .expect("valid 3-qubit marked probability for marks [2, 6]");
        assert!(mass > 0.9, "mass={mass}");
    }

    #[test]
    fn no_marked_items_zero_iterations() {
        // Empty oracle ⇒ no amplification, uniform state, argmax index 0.
        let res = grover_search(&[], 3).expect("valid 3-qubit grover search with empty oracle");
        assert_eq!(res.iterations, 0);
        // Uniform state: every amplitude is 1/√8, probability 1/8.
        assert!(
            (res.probability - 1.0 / 8.0).abs() < 1e-5,
            "p={}",
            res.probability
        );
    }

    #[test]
    fn n_qubits_zero_errors() {
        assert!(grover_search(&[0], 0).is_err());
    }

    #[test]
    fn probability_concentrates_with_more_qubits() {
        // Larger spaces still concentrate >0.9 on the single marked item.
        for n in 2..=6 {
            let target = (1usize << n) - 1; // last index
            let mass = marked_probability(&[target], n)
                .expect("valid marked probability for last-index target");
            assert!(mass > 0.8, "n={n} mass={mass}");
        }
    }

    #[test]
    fn deterministic_runs() {
        // Argmax readout has no randomness: identical runs agree exactly.
        let a = grover_search(&[3], 4).expect("valid 4-qubit grover search for mark 3 (run a)");
        let b = grover_search(&[3], 4).expect("valid 4-qubit grover search for mark 3 (run b)");
        assert_eq!(a.measured_index, b.measured_index);
        assert!((a.probability - b.probability).abs() < 1e-7);
        assert_eq!(a.iterations, b.iterations);
    }

    #[test]
    fn single_item_space() {
        // Smallest space where Grover yields certainty: n=2 (N=4), single mark.
        // Exactly one iteration rotates the state onto the marked item (prob 1).
        // (n=1 / N=2 is the known degenerate case where amplification cannot
        // exceed 1/2, so the meaningful "small space" check uses N=4.)
        let res = grover_search(&[2], 2).expect("valid 2-qubit grover search for mark 2");
        assert_eq!(res.measured_index, 2, "got {}", res.measured_index);
        assert!(res.probability > 0.99, "prob={}", res.probability);
    }

    #[test]
    fn n_eq_one_is_degenerate() {
        // N=2, M=1: Grover provably cannot amplify beyond the uniform 1/2.
        // The routine must still run cleanly and report a valid probability.
        let mass =
            marked_probability(&[1], 1).expect("valid 1-qubit marked probability for single mark");
        assert!((mass - 0.5).abs() < 1e-4, "mass={mass}");
    }

    #[test]
    fn out_of_range_mark_errors() {
        // Index 8 is out of range for a 3-qubit (dim 8 ⇒ valid 0..=7) space.
        assert!(grover_search(&[8], 3).is_err());
    }

    #[test]
    fn preserves_norm_through_iterations() {
        // Grover iterates are unitary; the marked + unmarked mass must sum to 1.
        let n = 4;
        let marks = [1usize, 7, 11];
        let marked = marked_probability(&marks, n)
            .expect("valid 4-qubit marked probability for marks [1, 7, 11]");
        assert!(marked <= 1.0 + 1e-5, "marked mass {marked} exceeds 1");
        // Re-derive total via a fresh full-state norm check.
        let mut all = vec![false; 1 << n];
        for &m in &marks {
            all[m] = true;
        }
        let n_marked = marks.len();
        let mut sv =
            StateVector::new_zero_state(n).expect("valid 4-qubit zero state for norm check");
        for q in 0..n {
            apply_1q_inplace(&mut sv, q, &gate_h()).expect("valid qubit index for Hadamard gate");
        }
        for _ in 0..optimal_iterations(1 << n, n_marked) {
            apply_oracle(&mut sv, &all);
            apply_diffusion(&mut sv);
        }
        assert!((sv.norm_sq() - 1.0).abs() < 1e-4, "norm={}", sv.norm_sq());
    }
}
