//! Superdense coding.
//!
//! Superdense coding is the dual of teleportation: it transmits **two** classical
//! bits by sending a **single** qubit, given a pre-shared Bell pair. Alice
//! encodes the two-bit message `(b0, b1)` into her half of the pair using one of
//! four Pauli operations, sends that one qubit to Bob, and Bob recovers both bits
//! by a Bell-basis measurement on the (now reunited) two qubits.
//!
//! Qubit layout (little-endian, two qubits):
//!
//! * **qubit 0** — Alice's half of the Bell pair (the qubit she physically sends).
//! * **qubit 1** — Bob's half of the Bell pair.
//!
//! Protocol:
//!
//! 1. Share the Bell pair `|Φ⁺⟩ = (|00⟩ + |11⟩)/√2` via `H₀ · CNOT₀→₁`.
//! 2. Alice's encoding on qubit 0, indexed by `(b0, b1)`:
//!    `(0,0) → I`, `(0,1) → X`, `(1,0) → Z`, `(1,1) → ZX` (i.e. `iY` up to phase),
//!    mapping `|Φ⁺⟩` onto the four orthogonal Bell states.
//! 3. Bob decodes with the inverse Bell circuit `CNOT₀→₁ · H₀`, then reads the
//!    two computational-basis bits, recovering `(b0, b1)` deterministically.

use crate::error::QuantumResult;
use crate::gates::controlled::apply_cnot;
use crate::gates::hadamard::gate_h;
use crate::gates::pauli::{gate_x, gate_z};
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Outcome of [`superdense_decode`].
#[derive(Debug, Clone)]
pub struct SuperdenseResult {
    /// The two decoded classical bits `(b0, b1)`.
    pub decoded: (bool, bool),
    /// Probability mass on the decoded computational-basis string (≈ 1).
    pub probability: f32,
}

/// Run the full superdense-coding round-trip for the message `(b0, b1)`.
///
/// Prepares a Bell pair, applies Alice's Pauli encoding for the requested bits,
/// then applies Bob's decoder and reads out both qubits via deterministic
/// argmax. The decoded bits equal the input `(b0, b1)` exactly.
///
/// # Errors
/// Propagates gate-application errors (none occur for the fixed 2-qubit circuit).
pub fn superdense_decode(b0: bool, b1: bool) -> QuantumResult<SuperdenseResult> {
    // Step 1: Bell pair |Φ⁺⟩ on qubits 0 and 1.
    let mut sv = StateVector::new_zero_state(2)?;
    apply_1q_inplace(&mut sv, 0, &gate_h())?;
    apply_cnot(&mut sv, 0, 1)?;

    // Step 2: Alice's encoding on qubit 0. Apply X for b1, Z for b0.
    // Order ZX (X first then Z) maps to the standard Bell-state assignment.
    if b1 {
        apply_1q_inplace(&mut sv, 0, &gate_x())?;
    }
    if b0 {
        apply_1q_inplace(&mut sv, 0, &gate_z())?;
    }

    // Step 3: Bob's decoder — inverse of the Bell-pair circuit.
    apply_cnot(&mut sv, 0, 1)?;
    apply_1q_inplace(&mut sv, 0, &gate_h())?;

    // Deterministic argmax readout of the 2-qubit register.
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, a) in sv.amps.iter().enumerate() {
        let p = a.norm_sqr();
        if p > best {
            best = p;
            best_idx = i;
        }
    }

    // Little-endian: qubit 0 holds the first decoded bit, qubit 1 the second.
    let decoded_b0 = (best_idx & 1) != 0;
    let decoded_b1 = (best_idx & 2) != 0;

    Ok(SuperdenseResult {
        decoded: (decoded_b0, decoded_b1),
        probability: best,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_zero_zero() {
        let res = superdense_decode(false, false)
            .expect("superdense decode of (false, false) should succeed");
        assert_eq!(res.decoded, (false, false));
        assert!(res.probability > 0.999, "p={}", res.probability);
    }

    #[test]
    fn decodes_zero_one() {
        let res = superdense_decode(false, true)
            .expect("superdense decode of (false, true) should succeed");
        assert_eq!(res.decoded, (false, true));
        assert!(res.probability > 0.999, "p={}", res.probability);
    }

    #[test]
    fn decodes_one_zero() {
        let res = superdense_decode(true, false)
            .expect("superdense decode of (true, false) should succeed");
        assert_eq!(res.decoded, (true, false));
        assert!(res.probability > 0.999, "p={}", res.probability);
    }

    #[test]
    fn decodes_one_one() {
        let res = superdense_decode(true, true)
            .expect("superdense decode of (true, true) should succeed");
        assert_eq!(res.decoded, (true, true));
        assert!(res.probability > 0.999, "p={}", res.probability);
    }

    #[test]
    fn all_messages_round_trip() {
        // Exhaustive: every two-bit message must round-trip exactly.
        for &(b0, b1) in &[(false, false), (false, true), (true, false), (true, true)] {
            let res = superdense_decode(b0, b1)
                .expect("superdense decode should succeed for all two-bit messages");
            assert_eq!(res.decoded, (b0, b1), "failed for ({b0},{b1})");
            assert!(res.probability > 0.999, "({b0},{b1}) p={}", res.probability);
        }
    }

    #[test]
    fn encodings_are_orthogonal() {
        // The four messages must decode to four *distinct* basis strings.
        let mut seen = std::collections::HashSet::new();
        for &(b0, b1) in &[(false, false), (false, true), (true, false), (true, true)] {
            let res = superdense_decode(b0, b1)
                .expect("superdense decode should succeed for orthogonality check");
            let idx = usize::from(res.decoded.0) | (usize::from(res.decoded.1) << 1);
            assert!(seen.insert(idx), "collision at message ({b0},{b1})");
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn probability_is_deterministic() {
        let a = superdense_decode(true, false)
            .expect("superdense decode (true, false) first call should succeed");
        let b = superdense_decode(true, false)
            .expect("superdense decode (true, false) second call should succeed");
        assert_eq!(a.decoded, b.decoded);
        assert!((a.probability - b.probability).abs() < 1e-7);
    }

    #[test]
    fn probability_finite_and_in_range() {
        let res = superdense_decode(true, true)
            .expect("superdense decode of (true, true) should succeed");
        assert!(res.probability.is_finite());
        assert!((0.0..=1.0 + 1e-5).contains(&res.probability));
    }

    #[test]
    fn decoded_state_is_normalized() {
        // The decode circuit is unitary; the post-decode norm stays 1.
        let mut sv = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("H gate on qubit 0 should succeed");
        apply_cnot(&mut sv, 0, 1).expect("CNOT on qubits 0 to 1 should succeed");
        apply_1q_inplace(&mut sv, 0, &gate_x()).expect("X gate on qubit 0 should succeed");
        apply_1q_inplace(&mut sv, 0, &gate_z()).expect("Z gate on qubit 0 should succeed");
        apply_cnot(&mut sv, 0, 1).expect("CNOT on qubits 0 to 1 (decode step) should succeed");
        apply_1q_inplace(&mut sv, 0, &gate_h())
            .expect("H gate on qubit 0 (decode step) should succeed");
        assert!((sv.norm_sq() - 1.0).abs() < 1e-5, "norm={}", sv.norm_sq());
    }
}
