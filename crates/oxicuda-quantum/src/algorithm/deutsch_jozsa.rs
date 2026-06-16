//! Deutsch–Jozsa algorithm.
//!
//! Given a promise that a Boolean function `f : {0,1}^n → {0,1}` is either
//! *constant* (same output for every input) or *balanced* (output `0` for
//! exactly half the inputs and `1` for the other half), the Deutsch–Jozsa
//! algorithm decides which case holds with a **single** oracle query —
//! exponentially fewer than the `2^{n-1}+1` classical queries needed in the
//! worst case for deterministic certainty.
//!
//! The circuit acts on `n` query qubits (the ancilla / phase-kickback target is
//! folded into the oracle as a relative sign, so no explicit ancilla is
//! simulated):
//!
//! 1. Prepare `H^{⊗n}|0⟩`, the uniform superposition.
//! 2. Apply the phase oracle `O_f|x⟩ = (-1)^{f(x)}|x⟩`.
//! 3. Apply `H^{⊗n}` again.
//! 4. Measure. The probability of reading `|0…0⟩` is `1` if `f` is constant and
//!    `0` if `f` is balanced.
//!
//! This implementation evaluates the user-supplied `f` over the truth table to
//! build the phase oracle directly on the amplitude vector, which is both exact
//! and `O(2^n)`.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::hadamard::gate_h;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Classification returned by [`deutsch_jozsa`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionKind {
    /// `f` outputs the same bit for every input.
    Constant,
    /// `f` outputs `0` for half the inputs and `1` for the other half.
    Balanced,
}

/// Outcome of [`deutsch_jozsa`].
#[derive(Debug, Clone)]
pub struct DeutschJozsaResult {
    /// Decision: constant vs. balanced.
    pub kind: FunctionKind,
    /// Probability of measuring the all-zeros string `|0…0⟩`.
    pub all_zero_probability: f32,
}

/// Run Deutsch–Jozsa for a function given as a truth-table closure.
///
/// `oracle(x)` must return `f(x) ∈ {false, true}` for every basis index
/// `x ∈ [0, 2^n)`. The routine returns [`FunctionKind::Constant`] when the
/// post-circuit all-zeros probability is `≈ 1`, and [`FunctionKind::Balanced`]
/// otherwise (the promise guarantees these are the only two possibilities).
///
/// # Errors
/// [`QuantumError::InvalidQubitCount`] if `n_qubits` is `0` or `> 30`.
pub fn deutsch_jozsa<F>(oracle: F, n_qubits: usize) -> QuantumResult<DeutschJozsaResult>
where
    F: Fn(usize) -> bool,
{
    if n_qubits == 0 || n_qubits > 30 {
        return Err(QuantumError::InvalidQubitCount { n: n_qubits });
    }

    // Step 1: uniform superposition.
    let mut sv = StateVector::new_zero_state(n_qubits)?;
    for q in 0..n_qubits {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    // Step 2: phase oracle (-1)^{f(x)} applied directly to each amplitude.
    for (x, a) in sv.amps.iter_mut().enumerate() {
        if oracle(x) {
            *a = -*a;
        }
    }

    // Step 3: re-apply Hadamards.
    for q in 0..n_qubits {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    // Step 4: probability of the all-zeros basis string.
    let all_zero_probability = sv.amps[0].norm_sqr();
    let kind = if all_zero_probability > 0.5 {
        FunctionKind::Constant
    } else {
        FunctionKind::Balanced
    };

    Ok(DeutschJozsaResult {
        kind,
        all_zero_probability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_zero_is_constant() {
        // f(x) = 0 for all x ⇒ all-zero probability 1.
        let res = deutsch_jozsa(|_| false, 3)
            .expect("n=3 is within [1, 30] and constant-zero oracle is well-defined");
        assert_eq!(res.kind, FunctionKind::Constant);
        assert!(
            (res.all_zero_probability - 1.0).abs() < 1e-5,
            "p={}",
            res.all_zero_probability
        );
    }

    #[test]
    fn constant_one_is_constant() {
        // f(x) = 1 for all x ⇒ global phase only, still all-zero probability 1.
        let res = deutsch_jozsa(|_| true, 4)
            .expect("n=4 is within [1, 30] and constant-one oracle is well-defined");
        assert_eq!(res.kind, FunctionKind::Constant);
        assert!(
            (res.all_zero_probability - 1.0).abs() < 1e-5,
            "p={}",
            res.all_zero_probability
        );
    }

    #[test]
    fn parity_function_is_balanced() {
        // f(x) = parity(x) is balanced ⇒ all-zero probability 0.
        let res = deutsch_jozsa(|x| (x.count_ones() & 1) == 1, 3)
            .expect("n=3 is within [1, 30] and parity is a well-defined boolean function");
        assert_eq!(res.kind, FunctionKind::Balanced);
        assert!(
            res.all_zero_probability < 1e-5,
            "p={}",
            res.all_zero_probability
        );
    }

    #[test]
    fn single_bit_balanced() {
        // f(x) = x_0 (least-significant bit) is balanced.
        let res = deutsch_jozsa(|x| (x & 1) == 1, 3)
            .expect("n=3 is within [1, 30] and the LSB oracle is well-defined");
        assert_eq!(res.kind, FunctionKind::Balanced);
        assert!(
            res.all_zero_probability < 1e-5,
            "p={}",
            res.all_zero_probability
        );
    }

    #[test]
    fn balanced_half_outputs_one() {
        // Mark exactly the upper half of a 3-qubit space (x >= 4): balanced.
        let res = deutsch_jozsa(|x| x >= 4, 3)
            .expect("n=3 is within [1, 30] and the threshold oracle is well-defined");
        assert_eq!(res.kind, FunctionKind::Balanced);
    }

    #[test]
    fn single_qubit_constant() {
        // n=1 constant function: all-zero probability 1.
        let res = deutsch_jozsa(|_| false, 1)
            .expect("n=1 is within [1, 30] and the constant-zero oracle is well-defined");
        assert_eq!(res.kind, FunctionKind::Constant);
        assert!((res.all_zero_probability - 1.0).abs() < 1e-5);
    }

    #[test]
    fn single_qubit_balanced() {
        // n=1, f(x)=x is balanced (f(0)=0, f(1)=1).
        let res = deutsch_jozsa(|x| x == 1, 1)
            .expect("n=1 is within [1, 30] and the identity oracle is well-defined");
        assert_eq!(res.kind, FunctionKind::Balanced);
        assert!(res.all_zero_probability < 1e-5);
    }

    #[test]
    fn n_qubits_zero_errors() {
        assert!(deutsch_jozsa(|_| false, 0).is_err());
    }

    #[test]
    fn probability_in_range_and_finite() {
        let res = deutsch_jozsa(|x| (x & 2) != 0, 4)
            .expect("n=4 is within [1, 30] and the bit-1 mask oracle is well-defined");
        assert!(res.all_zero_probability.is_finite());
        assert!(
            (0.0..=1.0 + 1e-5).contains(&res.all_zero_probability),
            "p={}",
            res.all_zero_probability
        );
    }

    #[test]
    fn deterministic() {
        let a = deutsch_jozsa(|x| (x & 1) == 1, 4)
            .expect("n=4 is within [1, 30] and the LSB oracle is well-defined");
        let b = deutsch_jozsa(|x| (x & 1) == 1, 4)
            .expect("n=4 is within [1, 30] and the LSB oracle is well-defined");
        assert_eq!(a.kind, b.kind);
        assert!((a.all_zero_probability - b.all_zero_probability).abs() < 1e-7);
    }
}
