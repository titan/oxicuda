//! Bernstein–Vazirani algorithm.
//!
//! Given an oracle for `f(x) = s · x (mod 2)` — the bitwise dot product of an
//! unknown `n`-bit hidden string `s` with the input `x` — Bernstein–Vazirani
//! recovers all `n` bits of `s` with a **single** quantum query, versus the `n`
//! queries required classically (one per basis vector `e_i`).
//!
//! The circuit is identical in shape to Deutsch–Jozsa:
//!
//! 1. Prepare `H^{⊗n}|0⟩`.
//! 2. Apply the phase oracle `O_f|x⟩ = (-1)^{s·x}|x⟩`.
//! 3. Apply `H^{⊗n}`.
//! 4. Measure — the register collapses deterministically to `|s⟩`.
//!
//! The key identity is `H^{⊗n} (-1)^{s·x} H^{⊗n}|0⟩ = |s⟩`: the phase pattern of
//! the oracle is exactly the Hadamard transform of the basis state `|s⟩`.
//!
//! With this crate's **little-endian** convention, bit `i` of `s` corresponds to
//! qubit `i` (mask `1 << i`), and the recovered integer satisfies
//! `s = Σ_i s_i · 2^i`.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::hadamard::gate_h;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Outcome of [`bernstein_vazirani`].
#[derive(Debug, Clone)]
pub struct BernsteinVaziraniResult {
    /// The recovered hidden integer `s` (little-endian: bit `i` ↦ qubit `i`).
    pub hidden_string: usize,
    /// Probability mass on the recovered basis state (≈ 1 for a perfect oracle).
    pub probability: f32,
}

/// Run Bernstein–Vazirani to recover the hidden string behind an `s · x` oracle.
///
/// `secret` is the hidden bit-string encoded as an integer in `[0, 2^n)`. The
/// routine builds the corresponding phase oracle, runs the circuit, and reads
/// the register back via deterministic argmax — which equals `secret` exactly.
///
/// # Errors
/// * [`QuantumError::InvalidQubitCount`] if `n_qubits` is `0` or `> 30`.
/// * [`QuantumError::InvalidParameter`] if `secret >= 2^n_qubits`.
pub fn bernstein_vazirani(
    secret: usize,
    n_qubits: usize,
) -> QuantumResult<BernsteinVaziraniResult> {
    if n_qubits == 0 || n_qubits > 30 {
        return Err(QuantumError::InvalidQubitCount { n: n_qubits });
    }
    let dim = 1usize << n_qubits;
    if secret >= dim {
        return Err(QuantumError::InvalidParameter {
            name: format!("secret {secret} out of range for {n_qubits} qubits"),
        });
    }

    // Step 1: uniform superposition.
    let mut sv = StateVector::new_zero_state(n_qubits)?;
    for q in 0..n_qubits {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    // Step 2: phase oracle (-1)^{s·x}. The dot product parity is the parity of
    // the population count of (s AND x).
    for (x, a) in sv.amps.iter_mut().enumerate() {
        if ((secret & x).count_ones() & 1) == 1 {
            *a = -*a;
        }
    }

    // Step 3: re-apply Hadamards; the state is now |secret⟩.
    for q in 0..n_qubits {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    // Step 4: deterministic argmax readout.
    let mut hidden_string = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, amp) in sv.amps.iter().enumerate() {
        let p = amp.norm_sqr();
        if p > best {
            best = p;
            hidden_string = i;
        }
    }

    Ok(BernsteinVaziraniResult {
        hidden_string,
        probability: best,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_all_zero_string() {
        // s = 0 ⇒ constant-zero oracle ⇒ recovered string 0.
        let res = bernstein_vazirani(0, 4).expect("secret=0 is within range for 4 qubits");
        assert_eq!(res.hidden_string, 0);
        assert!(res.probability > 0.999, "p={}", res.probability);
    }

    #[test]
    fn recovers_all_one_string() {
        // s = 0b1111 = 15 on 4 qubits.
        let res = bernstein_vazirani(15, 4).expect("secret=15 is within range for 4 qubits");
        assert_eq!(res.hidden_string, 15);
        assert!(res.probability > 0.999, "p={}", res.probability);
    }

    #[test]
    fn recovers_arbitrary_string() {
        // s = 0b1011 = 11.
        let res =
            bernstein_vazirani(0b1011, 4).expect("secret=0b1011 is within range for 4 qubits");
        assert_eq!(res.hidden_string, 0b1011);
        assert!(res.probability > 0.999, "p={}", res.probability);
    }

    #[test]
    fn recovers_every_string_for_n3() {
        // Exhaustively confirm recovery across the whole 3-qubit secret space.
        for s in 0..8usize {
            let res = bernstein_vazirani(s, 3).expect("s is in 0..8, within range for 3 qubits");
            assert_eq!(res.hidden_string, s, "failed for s={s}");
            assert!(res.probability > 0.999, "s={s} p={}", res.probability);
        }
    }

    #[test]
    fn single_qubit_recovers_bit() {
        let r0 = bernstein_vazirani(0, 1).expect("secret=0 is within range for 1 qubit");
        assert_eq!(r0.hidden_string, 0);
        let r1 = bernstein_vazirani(1, 1).expect("secret=1 is within range for 1 qubit");
        assert_eq!(r1.hidden_string, 1);
    }

    #[test]
    fn probability_concentrates() {
        // The recovered basis state holds essentially all the probability.
        let res =
            bernstein_vazirani(0b101010, 6).expect("secret=0b101010 is within range for 6 qubits");
        assert_eq!(res.hidden_string, 0b101010);
        assert!(res.probability > 0.999, "p={}", res.probability);
    }

    #[test]
    fn n_qubits_zero_errors() {
        assert!(bernstein_vazirani(0, 0).is_err());
    }

    #[test]
    fn secret_out_of_range_errors() {
        // 8 needs 4 bits; with 3 qubits the valid range is 0..=7.
        assert!(bernstein_vazirani(8, 3).is_err());
    }

    #[test]
    fn probability_finite_and_in_range() {
        let res = bernstein_vazirani(5, 4).expect("secret=5 is within range for 4 qubits");
        assert!(res.probability.is_finite());
        assert!((0.0..=1.0 + 1e-5).contains(&res.probability));
    }

    #[test]
    fn deterministic() {
        let a = bernstein_vazirani(9, 4).expect("secret=9 is within range for 4 qubits");
        let b = bernstein_vazirani(9, 4)
            .expect("secret=9 is within range for 4 qubits, second call for determinism check");
        assert_eq!(a.hidden_string, b.hidden_string);
        assert!((a.probability - b.probability).abs() < 1e-7);
    }
}
