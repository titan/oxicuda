//! Multi-qubit entangled-state preparation and 3-qubit quantum error correction.
//!
//! This module collects two closely related primitives that underpin
//! fault-tolerant quantum computing demos:
//!
//! ## Entangled state preparation
//! * [`prepare_ghz`] — the `n`-qubit Greenberger–Horne–Zeilinger state
//!   `(|0…0⟩ + |1…1⟩)/√2`, built from one Hadamard plus a CNOT ladder.
//! * [`prepare_w`] — the `n`-qubit W state, the equal superposition of all
//!   single-excitation basis vectors `Σ_i |0…1_i…0⟩ / √n`, built from a cascade
//!   of controlled `Y`-rotations.
//!
//! ## 3-qubit repetition codes
//! * [`bit_flip_encode`] / [`bit_flip_correct`] — protect against a single `X`
//!   error by majority vote over the syndrome `{Z₀Z₁, Z₁Z₂}`.
//! * [`phase_flip_encode`] / [`phase_flip_correct`] — the Hadamard-conjugated
//!   code protecting against a single `Z` error.
//!
//! All routines act on this crate's **little-endian** [`StateVector`] (qubit `i`
//! ↦ bit `i`), so the logical qubit is encoded across physical qubits
//! `0, 1, 2`.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::controlled::apply_cnot;
use crate::gates::hadamard::gate_h;
use crate::gates::parametric::gate_ry;
use crate::statevec::apply_1q::{apply_1q_controlled, apply_1q_inplace};
use crate::statevec::state::StateVector;

/// Prepare the `n`-qubit GHZ state `(|0…0⟩ + |1…1⟩)/√2`.
///
/// # Errors
/// [`QuantumError::InvalidQubitCount`] if `n_qubits` is `0` or `> 30`.
pub fn prepare_ghz(n_qubits: usize) -> QuantumResult<StateVector> {
    if n_qubits == 0 || n_qubits > 30 {
        return Err(QuantumError::InvalidQubitCount { n: n_qubits });
    }
    let mut sv = StateVector::new_zero_state(n_qubits)?;
    apply_1q_inplace(&mut sv, 0, &gate_h())?;
    for q in 1..n_qubits {
        apply_cnot(&mut sv, q - 1, q)?;
    }
    Ok(sv)
}

/// Prepare the `n`-qubit W state `Σ_{i} |0…1_i…0⟩ / √n`.
///
/// Construction (Cabello 2002 cascade): qubit `0` is rotated so that the
/// excitation either stays on qubit `0` with amplitude `1/√n` or is forwarded.
/// Successive controlled `Y`-rotations split the remaining amplitude evenly, and
/// CNOTs move the excitation down the chain. The net result places exactly one
/// `|1⟩` with equal amplitude on every qubit.
///
/// # Errors
/// [`QuantumError::InvalidQubitCount`] if `n_qubits` is `0` or `> 30`.
pub fn prepare_w(n_qubits: usize) -> QuantumResult<StateVector> {
    if n_qubits == 0 || n_qubits > 30 {
        return Err(QuantumError::InvalidQubitCount { n: n_qubits });
    }
    let mut sv = StateVector::new_zero_state(n_qubits)?;
    if n_qubits == 1 {
        apply_1q_inplace(&mut sv, 0, &crate::gates::pauli::gate_x())?;
        return Ok(sv);
    }
    // Put the excitation on qubit 0 first (X), then peel amplitude off toward
    // higher qubits with controlled rotations.
    apply_1q_inplace(&mut sv, 0, &crate::gates::pauli::gate_x())?;
    // For qubit k = 0..n-2: rotate qubit k+1 conditioned on qubit k so that a
    // fraction 1/(n-k) of the population remains on qubit k.
    for k in 0..n_qubits - 1 {
        let remaining = (n_qubits - k) as f32;
        // Want P(stay on k) = 1/remaining ⇒ split angle θ with cos²(θ/2)=1/remaining.
        let theta = 2.0 * (1.0 / remaining).sqrt().acos();
        // Controlled-Ry(θ) with control = qubit k, target = qubit k+1.
        apply_1q_controlled(&mut sv, k, k + 1, &gate_ry(theta))?;
        // Move excitation: if qubit k+1 became 1, clear qubit k via CNOT(k+1 → k).
        apply_cnot(&mut sv, k + 1, k)?;
    }
    Ok(sv)
}

/// Encode a single logical qubit (state `α|0⟩ + β|1⟩` on physical qubit `0`)
/// into the 3-qubit **bit-flip** repetition code `α|000⟩ + β|111⟩`.
///
/// The input state vector must have exactly 3 qubits with qubits 1 and 2 in
/// `|0⟩` (the standard ancilla initialisation). Two CNOTs copy the logical bit.
///
/// # Errors
/// [`QuantumError::InvalidQubitCount`] if the state does not have 3 qubits.
pub fn bit_flip_encode(state: &mut StateVector) -> QuantumResult<()> {
    if state.n_qubits != 3 {
        return Err(QuantumError::InvalidQubitCount { n: state.n_qubits });
    }
    apply_cnot(state, 0, 1)?;
    apply_cnot(state, 0, 2)?;
    Ok(())
}

/// Measure the bit-flip syndrome `(Z₀Z₁, Z₁Z₂)` *coherently* via parity over the
/// amplitude array, returning the two parity bits. A deterministic state yields a
/// unique syndrome; superpositions of distinct syndromes return the dominant one.
fn bit_flip_syndrome(state: &StateVector) -> (bool, bool) {
    // s0 = parity(bit0 XOR bit1), s1 = parity(bit1 XOR bit2), weighted by prob.
    let mut p_s0 = [0.0f32; 2];
    let mut p_s1 = [0.0f32; 2];
    for (idx, amp) in state.amps.iter().enumerate() {
        let pr = amp.norm_sqr();
        if pr == 0.0 {
            continue;
        }
        let b0 = idx & 1;
        let b1 = (idx >> 1) & 1;
        let b2 = (idx >> 2) & 1;
        p_s0[b0 ^ b1] += pr;
        p_s1[b1 ^ b2] += pr;
    }
    (p_s0[1] > p_s0[0], p_s1[1] > p_s1[0])
}

/// Correct a single bit-flip error using the syndrome majority vote and return
/// the index of the qubit that was corrected (`None` if no error detected).
///
/// Syndrome decoding:
/// * `(0,0)` → no error.
/// * `(1,0)` → error on qubit 0.
/// * `(1,1)` → error on qubit 1.
/// * `(0,1)` → error on qubit 2.
///
/// # Errors
/// [`QuantumError::InvalidQubitCount`] if the state does not have 3 qubits.
pub fn bit_flip_correct(state: &mut StateVector) -> QuantumResult<Option<usize>> {
    if state.n_qubits != 3 {
        return Err(QuantumError::InvalidQubitCount { n: state.n_qubits });
    }
    let (s0, s1) = bit_flip_syndrome(state);
    let correct_qubit = match (s0, s1) {
        (false, false) => None,
        (true, false) => Some(0),
        (true, true) => Some(1),
        (false, true) => Some(2),
    };
    if let Some(q) = correct_qubit {
        apply_1q_inplace(state, q, &crate::gates::pauli::gate_x())?;
    }
    Ok(correct_qubit)
}

/// Encode a single logical qubit into the 3-qubit **phase-flip** repetition code
/// `α|+++⟩ + β|−−−⟩`.
///
/// This is the bit-flip code conjugated by Hadamards: encode in the
/// computational basis, then map each qubit into the `±` (Hadamard) basis.
///
/// # Errors
/// [`QuantumError::InvalidQubitCount`] if the state does not have 3 qubits.
pub fn phase_flip_encode(state: &mut StateVector) -> QuantumResult<()> {
    bit_flip_encode(state)?;
    for q in 0..3 {
        apply_1q_inplace(state, q, &gate_h())?;
    }
    Ok(())
}

/// Correct a single phase-flip error by rotating back to the computational
/// basis, running the bit-flip decoder, and returning to the `±` basis.
///
/// Returns the index of the corrected qubit (`None` if no error detected).
///
/// # Errors
/// [`QuantumError::InvalidQubitCount`] if the state does not have 3 qubits.
pub fn phase_flip_correct(state: &mut StateVector) -> QuantumResult<Option<usize>> {
    if state.n_qubits != 3 {
        return Err(QuantumError::InvalidQubitCount { n: state.n_qubits });
    }
    // Map ± basis → computational basis.
    for q in 0..3 {
        apply_1q_inplace(state, q, &gate_h())?;
    }
    let corrected = bit_flip_correct(state)?;
    // Map back to ± basis.
    for q in 0..3 {
        apply_1q_inplace(state, q, &gate_h())?;
    }
    Ok(corrected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::pauli::{gate_x, gate_z};

    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn ghz_two_qubit_is_bell() {
        let sv = prepare_ghz(2).expect("valid 2-qubit GHZ state");
        assert!((sv.amps[0].re - INV_SQRT2).abs() < 1e-5);
        assert!((sv.amps[3].re - INV_SQRT2).abs() < 1e-5);
        assert!(sv.amps[1].norm() < 1e-5);
        assert!(sv.amps[2].norm() < 1e-5);
    }

    #[test]
    fn ghz_three_qubit_amplitudes() {
        let sv = prepare_ghz(3).expect("valid 3-qubit GHZ state");
        assert!((sv.amps[0].re - INV_SQRT2).abs() < 1e-5, "|000>");
        assert!((sv.amps[7].re - INV_SQRT2).abs() < 1e-5, "|111>");
        // All other amplitudes vanish.
        for i in 1..7 {
            assert!(sv.amps[i].norm() < 1e-5, "amp[{i}]={:?}", sv.amps[i]);
        }
    }

    #[test]
    fn ghz_norm_preserved() {
        for n in 1..=5 {
            let sv = prepare_ghz(n).expect("valid n-qubit GHZ state");
            assert!((sv.norm_sq() - 1.0).abs() < 1e-4, "n={n}");
        }
    }

    #[test]
    fn ghz_invalid_qubit_count() {
        assert!(prepare_ghz(0).is_err());
        assert!(prepare_ghz(31).is_err());
    }

    #[test]
    fn w_state_single_excitation_amplitudes() {
        let sv = prepare_w(3).expect("valid 3-qubit W state");
        let amp = (1.0f32 / 3.0).sqrt();
        // |001>, |010>, |100> at indices 1,2,4.
        for &i in &[1usize, 2, 4] {
            assert!(
                (sv.amps[i].norm() - amp).abs() < 1e-4,
                "amp[{i}]={:?}",
                sv.amps[i]
            );
        }
        // Zero / multi excitation amplitudes vanish.
        for &i in &[0usize, 3, 5, 6, 7] {
            assert!(sv.amps[i].norm() < 1e-4, "amp[{i}]={:?}", sv.amps[i]);
        }
    }

    #[test]
    fn w_state_norm_is_one() {
        for n in 1..=5 {
            let sv = prepare_w(n).expect("valid n-qubit W state");
            assert!((sv.norm_sq() - 1.0).abs() < 1e-4, "n={n}");
        }
    }

    #[test]
    fn w_state_exactly_one_excitation() {
        // Every populated basis state has Hamming weight 1.
        let sv = prepare_w(4).expect("valid 4-qubit W state");
        for (idx, amp) in sv.amps.iter().enumerate() {
            if amp.norm() > 1e-4 {
                assert_eq!(idx.count_ones(), 1, "idx={idx:b} has weight ≠ 1");
            }
        }
    }

    #[test]
    fn w_state_uniform_marginals() {
        // Each qubit has equal probability 1/n of being |1>.
        let n = 4;
        let sv = prepare_w(n).expect("valid 4-qubit W state");
        for q in 0..n {
            let p1 = sv
                .measure_prob(q, true)
                .expect("valid qubit index for probability measurement");
            assert!((p1 - 1.0 / n as f32).abs() < 1e-4, "q={q} p1={p1}");
        }
    }

    #[test]
    fn bit_flip_encode_logical_one() {
        // |1> on qubit 0 → |111>.
        let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_x()).expect("X gate on qubit 0 succeeds");
        bit_flip_encode(&mut sv).expect("bit-flip encoding succeeds on 3-qubit state");
        assert!((sv.amps[7].norm() - 1.0).abs() < 1e-5, "expected |111>");
    }

    #[test]
    fn bit_flip_corrects_single_error_on_each_qubit() {
        for err_q in 0..3 {
            // Encode logical |1> = |111>.
            let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
            apply_1q_inplace(&mut sv, 0, &gate_x()).expect("X gate on qubit 0 succeeds");
            bit_flip_encode(&mut sv).expect("bit-flip encoding succeeds on 3-qubit state");
            // Inject X error.
            apply_1q_inplace(&mut sv, err_q, &gate_x())
                .expect("X error injection on target qubit succeeds");
            // Correct.
            let corrected = bit_flip_correct(&mut sv).expect("bit-flip correction succeeds");
            assert_eq!(corrected, Some(err_q), "error on {err_q}");
            // Back to |111>.
            assert!((sv.amps[7].norm() - 1.0).abs() < 1e-5, "err_q={err_q}");
        }
    }

    #[test]
    fn bit_flip_no_error_detected() {
        let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        bit_flip_encode(&mut sv).expect("bit-flip encoding of |000> succeeds"); // |000>
        let corrected =
            bit_flip_correct(&mut sv).expect("bit-flip correction on clean state succeeds");
        assert_eq!(corrected, None);
        assert!((sv.amps[0].norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn bit_flip_preserves_superposition_after_correction() {
        // Logical |+_L> = (|000> + |111>)/√2, error on qubit 1, then correct.
        let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("Hadamard gate on qubit 0 succeeds");
        bit_flip_encode(&mut sv).expect("bit-flip encoding of superposition succeeds");
        let before0 = sv.amps[0];
        let before7 = sv.amps[7];
        apply_1q_inplace(&mut sv, 1, &gate_x()).expect("X error injection on qubit 1 succeeds");
        let corrected =
            bit_flip_correct(&mut sv).expect("bit-flip correction restores superposition");
        assert_eq!(corrected, Some(1));
        assert!((sv.amps[0] - before0).norm() < 1e-5);
        assert!((sv.amps[7] - before7).norm() < 1e-5);
    }

    #[test]
    fn bit_flip_wrong_qubit_count_errors() {
        let mut sv = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        assert!(bit_flip_encode(&mut sv).is_err());
        assert!(bit_flip_correct(&mut sv).is_err());
    }

    #[test]
    fn phase_flip_corrects_single_z_error() {
        for err_q in 0..3 {
            // Encode logical |1>.
            let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
            apply_1q_inplace(&mut sv, 0, &gate_x()).expect("X gate on qubit 0 succeeds");
            phase_flip_encode(&mut sv).expect("phase-flip encoding succeeds on 3-qubit state");
            let encoded = sv.clone();
            // Inject Z error.
            apply_1q_inplace(&mut sv, err_q, &gate_z())
                .expect("Z error injection on target qubit succeeds");
            // Correct.
            let corrected = phase_flip_correct(&mut sv).expect("phase-flip correction succeeds");
            assert_eq!(corrected, Some(err_q), "z-error on {err_q}");
            // State restored to the encoded logical |1>.
            for i in 0..8 {
                assert!(
                    (sv.amps[i] - encoded.amps[i]).norm() < 1e-4,
                    "err_q={err_q} amp[{i}]"
                );
            }
        }
    }

    #[test]
    fn phase_flip_no_error_detected() {
        let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        phase_flip_encode(&mut sv).expect("phase-flip encoding succeeds");
        let corrected =
            phase_flip_correct(&mut sv).expect("phase-flip correction on clean state succeeds");
        assert_eq!(corrected, None);
    }
}
