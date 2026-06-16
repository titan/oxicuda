//! Quantum teleportation.
//!
//! Teleportation transmits an unknown single-qubit state `|ψ⟩ = α|0⟩ + β|1⟩`
//! from Alice to Bob using one pre-shared Bell pair and two classical bits, with
//! no quantum channel for `|ψ⟩` itself. The protocol famously **destroys** the
//! source copy (consistent with no-cloning) and reconstructs `|ψ⟩` on Bob's
//! qubit after two Pauli corrections conditioned on Alice's measurement.
//!
//! Qubit layout (little-endian, three qubits total):
//!
//! * **qubit 0** — Alice's message qubit, prepared in `|ψ⟩`.
//! * **qubit 1** — Alice's half of the Bell pair.
//! * **qubit 2** — Bob's half of the Bell pair (the teleportation destination).
//!
//! Protocol:
//!
//! 1. Entangle qubits 1 and 2 into a Bell pair: `H₁ · CNOT₁→₂`.
//! 2. Alice's Bell-basis measurement on (0,1): `CNOT₀→₁ · H₀`, then measure both.
//! 3. Send the two classical bits `(m0, m1)` to Bob.
//! 4. Bob corrects: apply `X` if `m1 = 1`, then `Z` if `m0 = 1`.
//!
//! After step 4 qubit 2 holds `|ψ⟩` exactly (up to floating-point round-off),
//! while qubits 0 and 1 are left in a definite computational-basis state.

use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::gates::controlled::apply_cnot;
use crate::gates::hadamard::gate_h;
use crate::gates::pauli::{gate_x, gate_z};
use crate::handle::LcgRng;
use crate::midcircuit::measurement::{ClassicalRegister, apply_if, measure_and_collapse};
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

/// Outcome of [`teleport`].
#[derive(Debug, Clone)]
pub struct TeleportResult {
    /// Alice's two measurement bits `(m0, m1)` (qubit 0 then qubit 1).
    pub classical_bits: (bool, bool),
    /// Bob's reconstructed amplitudes `(⟨0|ψ'⟩, ⟨1|ψ'⟩)` on qubit 2, obtained by
    /// projecting the post-correction three-qubit state onto Alice's measured
    /// branch and tracing out qubits 0 and 1.
    pub bob_amplitudes: (Complex32, Complex32),
    /// Fidelity `|⟨ψ|ψ'⟩|²` between the input state and Bob's output.
    pub fidelity: f32,
}

/// Teleport the single-qubit state `(alpha, beta)` from qubit 0 to qubit 2.
///
/// `alpha` and `beta` are the input amplitudes of `|ψ⟩ = α|0⟩ + β|1⟩`; they are
/// renormalized internally, so any non-zero pair is accepted. `rng` drives
/// Alice's measurement sampling. The returned [`TeleportResult`] reports the
/// measured classical bits, Bob's reconstructed amplitudes, and the fidelity
/// with the input (which is `≈ 1` for a correct protocol regardless of which
/// measurement branch occurred).
///
/// # Errors
/// [`QuantumError::InvalidParameter`] if both `alpha` and `beta` are (near) zero.
pub fn teleport(
    alpha: Complex32,
    beta: Complex32,
    rng: &mut LcgRng,
) -> QuantumResult<TeleportResult> {
    let norm = (alpha.norm_sqr() + beta.norm_sqr()).sqrt();
    if norm < 1e-12 {
        return Err(QuantumError::InvalidParameter {
            name: "input state amplitudes are both zero".into(),
        });
    }
    let a = alpha / norm;
    let b = beta / norm;

    // Build the 3-qubit state with qubit 0 = |ψ⟩, qubits 1,2 = |0⟩.
    // Little-endian: index bit q corresponds to qubit q.
    let mut amps = vec![Complex32::new(0.0, 0.0); 8];
    amps[0b000] = a; // qubit 0 = 0
    amps[0b001] = b; // qubit 0 = 1
    let mut sv = StateVector { amps, n_qubits: 3 };

    // Step 1: Bell pair on qubits 1 and 2.
    apply_1q_inplace(&mut sv, 1, &gate_h())?;
    apply_cnot(&mut sv, 1, 2)?;

    // Step 2: Alice's Bell-basis measurement basis change on (0,1).
    apply_cnot(&mut sv, 0, 1)?;
    apply_1q_inplace(&mut sv, 0, &gate_h())?;

    // Measure qubits 0 and 1, collapsing the state.
    let mut creg = ClassicalRegister::new(2);
    let m0 = measure_and_collapse(&mut sv, 0, rng, &mut creg, 0)?;
    let m1 = measure_and_collapse(&mut sv, 1, rng, &mut creg, 1)?;

    // Step 4: Bob's corrections — X if m1, then Z if m0.
    apply_if(&mut sv, &creg, &[(1, true)], &gate_x(), 2)?;
    apply_if(&mut sv, &creg, &[(0, true)], &gate_z(), 2)?;

    // Extract Bob's amplitudes. After collapse qubits 0,1 are fixed to (m0,m1);
    // the only non-zero amplitudes are at indices with those bits set.
    let base = (usize::from(m0)) | (usize::from(m1) << 1);
    let bob0 = sv.amps[base]; // qubit 2 = 0
    let bob1 = sv.amps[base | (1 << 2)]; // qubit 2 = 1

    // Fidelity |⟨ψ|ψ'⟩|² with the (normalized) input.
    let overlap = a.conj() * bob0 + b.conj() * bob1;
    let fidelity = overlap.norm_sqr();

    Ok(TeleportResult {
        classical_bits: (m0, m1),
        bob_amplitudes: (bob0, bob1),
        fidelity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn teleports_zero_state() {
        // |ψ⟩ = |0⟩ ⇒ Bob recovers |0⟩ with fidelity 1.
        let mut rng = LcgRng::new(1);
        let res = teleport(Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0), &mut rng)
            .expect("teleportation of |0⟩ state with normalized input");
        assert!(res.fidelity > 0.999, "fidelity={}", res.fidelity);
        assert!((res.bob_amplitudes.0.norm() - 1.0).abs() < 1e-4);
        assert!(res.bob_amplitudes.1.norm() < 1e-4);
    }

    #[test]
    fn teleports_one_state() {
        // |ψ⟩ = |1⟩ ⇒ Bob recovers |1⟩.
        let mut rng = LcgRng::new(2);
        let res = teleport(Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0), &mut rng)
            .expect("non-zero |1⟩ state is a valid teleportation input");
        assert!(res.fidelity > 0.999, "fidelity={}", res.fidelity);
        assert!(res.bob_amplitudes.0.norm() < 1e-4);
        assert!((res.bob_amplitudes.1.norm() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn teleports_plus_state() {
        // |ψ⟩ = |+⟩ = (|0⟩+|1⟩)/√2.
        let mut rng = LcgRng::new(3);
        let res = teleport(
            Complex32::new(FRAC_1_SQRT_2, 0.0),
            Complex32::new(FRAC_1_SQRT_2, 0.0),
            &mut rng,
        )
        .expect("normalized |+⟩ state is a valid teleportation input");
        assert!(res.fidelity > 0.999, "fidelity={}", res.fidelity);
    }

    #[test]
    fn teleports_complex_superposition() {
        // |ψ⟩ with a relative complex phase; fidelity must still be 1.
        let mut rng = LcgRng::new(4);
        let res = teleport(Complex32::new(0.6, 0.0), Complex32::new(0.0, 0.8), &mut rng)
            .expect("non-zero complex superposition is a valid teleportation input");
        assert!(res.fidelity > 0.999, "fidelity={}", res.fidelity);
    }

    #[test]
    fn fidelity_high_across_many_seeds() {
        // All four measurement branches should give fidelity ≈ 1.
        for seed in 0..32u64 {
            let mut rng = LcgRng::new(seed);
            let res = teleport(Complex32::new(0.6, 0.0), Complex32::new(0.8, 0.0), &mut rng)
                .expect("non-zero (0.6, 0.8) real state is a valid teleportation input");
            assert!(res.fidelity > 0.99, "seed={seed} fidelity={}", res.fidelity);
        }
    }

    #[test]
    fn unnormalized_input_is_accepted() {
        // Input (2, 0) renormalizes to |0⟩; teleportation still succeeds.
        let mut rng = LcgRng::new(5);
        let res = teleport(Complex32::new(2.0, 0.0), Complex32::new(0.0, 0.0), &mut rng)
            .expect("unnormalized (2, 0) is non-zero and renormalized internally");
        assert!(res.fidelity > 0.999, "fidelity={}", res.fidelity);
    }

    #[test]
    fn zero_input_errors() {
        let mut rng = LcgRng::new(6);
        assert!(teleport(Complex32::new(0.0, 0.0), Complex32::new(0.0, 0.0), &mut rng).is_err());
    }

    #[test]
    fn classical_bits_are_booleans() {
        // Sanity: both bits are produced (no panic) and fidelity holds.
        let mut rng = LcgRng::new(7);
        let res = teleport(
            Complex32::new(0.5, 0.5),
            Complex32::new(0.5, -0.5),
            &mut rng,
        )
        .expect("complex superposition with non-zero norm is a valid teleportation input");
        let (m0, m1) = res.classical_bits;
        // Compiler-checked booleans; just exercise them.
        let _ = m0 & m1;
        assert!(res.fidelity > 0.99, "fidelity={}", res.fidelity);
    }

    #[test]
    fn bob_amplitudes_match_input_up_to_branch() {
        // Bob's recovered amplitudes equal the input (renormalized) amplitudes.
        let mut rng = LcgRng::new(11);
        let a = Complex32::new(0.6, 0.0);
        let b = Complex32::new(0.0, 0.8);
        let res = teleport(a, b, &mut rng)
            .expect("non-zero (0.6, 0.8i) state is a valid teleportation input");
        assert!(
            (res.bob_amplitudes.0 - a).norm() < 1e-4,
            "bob0={:?}",
            res.bob_amplitudes.0
        );
        assert!(
            (res.bob_amplitudes.1 - b).norm() < 1e-4,
            "bob1={:?}",
            res.bob_amplitudes.1
        );
    }

    #[test]
    fn fidelity_is_finite_and_in_range() {
        let mut rng = LcgRng::new(13);
        let res = teleport(Complex32::new(0.6, 0.2), Complex32::new(0.3, 0.7), &mut rng)
            .expect("non-zero complex state (0.6+0.2i, 0.3+0.7i) is a valid teleportation input");
        assert!(res.fidelity.is_finite());
        assert!((0.0..=1.0 + 1e-4).contains(&res.fidelity));
    }
}
