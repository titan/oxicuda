//! Ancilla-interference primitives: the **SWAP test** and the **Hadamard test**.
//!
//! Both extract a number from the interference of an ancilla qubit prepared in
//! `|+⟩`, conditionally coupled to a work register, and measured in the `X`
//! basis. They are the canonical building blocks for overlap estimation and for
//! measuring expectation values of (controlled) unitaries.
//!
//! * [`swap_test`] estimates the squared state overlap `|⟨ψ|φ⟩|²` from the
//!   ancilla statistics `P(0) = (1 + |⟨ψ|φ⟩|²)/2` (Buhrman et al., 2001).
//! * [`hadamard_test`] estimates `Re⟨ψ|U|ψ⟩` (or `Im⟨ψ|U|ψ⟩` with the imaginary
//!   variant) from `P(0) = (1 ± value)/2`.
//!
//! Because this crate stores the full state vector, the ancilla measurement
//! probability is read off *exactly* (`measure_prob`) rather than sampled, so
//! the returned values are the analytic results, not shot estimates.

use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::gates::hadamard::{gate_h, gate_sdg};
use crate::statevec::apply_1q::{apply_1q_controlled, apply_1q_inplace};
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

/// Controlled-SWAP (Fredkin): swap qubits `q1` and `q2` when `ctrl` is `|1⟩`.
///
/// Implemented directly on the amplitude array — for every basis state with
/// `ctrl = 1` and differing `q1`/`q2` bits, the amplitude is exchanged with its
/// bit-swapped partner. This is exact and phase-free (unlike a `T`-decomposed
/// Toffoli network), which matters for entangled inputs.
fn apply_cswap(sv: &mut StateVector, ctrl: usize, q1: usize, q2: usize) -> QuantumResult<()> {
    let n = sv.n_qubits;
    for &qq in &[ctrl, q1, q2] {
        if qq >= n {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: qq,
                n_qubits: n,
            });
        }
    }
    if ctrl == q1 || ctrl == q2 || q1 == q2 {
        return Err(QuantumError::InvalidParameter {
            name: "ctrl, q1, q2 must be distinct".into(),
        });
    }
    let cm = 1usize << ctrl;
    let m1 = 1usize << q1;
    let m2 = 1usize << q2;
    let dim = sv.amps.len();
    for i in 0..dim {
        // Touch each (q1=1,q2=0) representative once and swap with its partner.
        if (i & cm) != 0 && (i & m1) != 0 && (i & m2) == 0 {
            let j = (i & !m1) | m2;
            sv.amps.swap(i, j);
        }
    }
    Ok(())
}

/// Build the product state `|0⟩_anc ⊗ |ψ1⟩ ⊗ |ψ2⟩` on `1 + 2k` qubits with the
/// ancilla as qubit `0`, register 1 on qubits `1..=k`, register 2 on
/// `k+1..=2k` (little-endian throughout).
fn tensor_with_ancilla(psi1: &StateVector, psi2: &StateVector) -> StateVector {
    let k = psi1.n_qubits;
    let n = 1 + 2 * k;
    let dim = 1usize << n;
    let mut amps = vec![Complex32::new(0.0, 0.0); dim];
    for (i1, &a1) in psi1.amps.iter().enumerate() {
        for (i2, &a2) in psi2.amps.iter().enumerate() {
            let global = (i1 << 1) | (i2 << (1 + k));
            amps[global] = a1 * a2;
        }
    }
    StateVector { amps, n_qubits: n }
}

/// Estimate the squared overlap `|⟨ψ1|ψ2⟩|²` of two `k`-qubit states via the
/// SWAP test.
///
/// The result is read from `P(ancilla = 0) = (1 + |⟨ψ1|ψ2⟩|²)/2` and clamped
/// to `[0, 1]` to absorb floating-point round-off.
///
/// # Errors
/// * [`QuantumError::DimensionMismatch`] if the two states act on different
///   numbers of qubits.
/// * [`QuantumError::InvalidQubitCount`] if the states are empty or the
///   combined register would exceed the simulator's qubit limit.
pub fn swap_test(psi1: &StateVector, psi2: &StateVector) -> QuantumResult<f32> {
    if psi1.n_qubits != psi2.n_qubits {
        return Err(QuantumError::DimensionMismatch {
            expected: psi1.n_qubits,
            got: psi2.n_qubits,
        });
    }
    let k = psi1.n_qubits;
    if k == 0 {
        return Err(QuantumError::InvalidQubitCount { n: 0 });
    }
    let n = 1 + 2 * k;
    if n > 30 {
        return Err(QuantumError::InvalidQubitCount { n });
    }

    let mut sv = tensor_with_ancilla(psi1, psi2);
    apply_1q_inplace(&mut sv, 0, &gate_h())?;
    for j in 0..k {
        apply_cswap(&mut sv, 0, 1 + j, 1 + k + j)?;
    }
    apply_1q_inplace(&mut sv, 0, &gate_h())?;

    let p0 = sv.measure_prob(0, false)?;
    Ok((2.0 * p0 - 1.0).clamp(0.0, 1.0))
}

/// Estimate `⟨ψ|U|ψ⟩` for a single-qubit state `ψ` and single-qubit gate `U`
/// via the Hadamard test.
///
/// With `imaginary = false` the real part `Re⟨ψ|U|ψ⟩` is returned (from
/// `P(0) = (1 + Re⟨ψ|U|ψ⟩)/2`); with `imaginary = true` the imaginary part
/// `Im⟨ψ|U|ψ⟩` is returned, obtained by inserting `S†` on the ancilla after the
/// first Hadamard.
///
/// # Errors
/// [`QuantumError::InvalidQubitCount`] if `psi` is not a single-qubit state.
pub fn hadamard_test(
    psi: &StateVector,
    gate: &[[Complex32; 2]; 2],
    imaginary: bool,
) -> QuantumResult<f32> {
    if psi.n_qubits != 1 {
        return Err(QuantumError::InvalidQubitCount { n: psi.n_qubits });
    }
    // |0⟩_anc ⊗ |ψ⟩ on 2 qubits: ancilla = qubit 0, system = qubit 1.
    let mut amps = vec![Complex32::new(0.0, 0.0); 4];
    amps[0] = psi.amps[0]; // |anc=0, sys=0⟩
    amps[2] = psi.amps[1]; // |anc=0, sys=1⟩
    let mut sv = StateVector { amps, n_qubits: 2 };

    apply_1q_inplace(&mut sv, 0, &gate_h())?;
    if imaginary {
        apply_1q_inplace(&mut sv, 0, &gate_sdg())?;
    }
    apply_1q_controlled(&mut sv, 0, 1, gate)?;
    apply_1q_inplace(&mut sv, 0, &gate_h())?;

    let p0 = sv.measure_prob(0, false)?;
    Ok(2.0 * p0 - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::controlled::apply_cnot;
    use crate::gates::hadamard::gate_s;
    use crate::gates::pauli::{gate_x, gate_z};

    fn ket_zero(n: usize) -> StateVector {
        StateVector::new_zero_state(n).expect("valid n-qubit zero state")
    }

    fn ket_one() -> StateVector {
        let mut sv = StateVector::new_zero_state(1).expect("valid 1-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_x()).expect("X gate applied to qubit 0");
        sv
    }

    fn ket_plus() -> StateVector {
        let mut sv = StateVector::new_zero_state(1).expect("valid 1-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("H gate applied to qubit 0");
        sv
    }

    #[test]
    fn swap_test_identical_states_overlap_one() {
        let a = ket_zero(1);
        let b = ket_zero(1);
        let ov = swap_test(&a, &b).expect("swap test on identical 1-qubit states");
        assert!((ov - 1.0).abs() < 1e-5, "overlap={ov}");
    }

    #[test]
    fn swap_test_orthogonal_states_overlap_zero() {
        let a = ket_zero(1);
        let b = ket_one();
        let ov = swap_test(&a, &b).expect("swap test on orthogonal 1-qubit states");
        assert!(ov.abs() < 1e-5, "overlap={ov}");
    }

    #[test]
    fn swap_test_zero_and_plus_overlap_half() {
        // |⟨0|+⟩|² = 1/2.
        let a = ket_zero(1);
        let b = ket_plus();
        let ov = swap_test(&a, &b).expect("swap test on |0⟩ and |+⟩ states");
        assert!((ov - 0.5).abs() < 1e-5, "overlap={ov}");
    }

    #[test]
    fn swap_test_two_qubit_identical_overlap_one() {
        // Bell-like 2-qubit state vs. itself.
        let mut a = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        apply_1q_inplace(&mut a, 0, &gate_h()).expect("H gate applied to qubit 0");
        apply_cnot(&mut a, 0, 1).expect("CNOT with control 0 target 1");
        let b = a.clone();
        let ov = swap_test(&a, &b).expect("swap test on identical 2-qubit Bell states");
        assert!((ov - 1.0).abs() < 1e-5, "overlap={ov}");
    }

    #[test]
    fn swap_test_two_qubit_orthogonal_overlap_zero() {
        let a = ket_zero(2); // |00⟩
        let mut b = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        apply_1q_inplace(&mut b, 1, &gate_x()).expect("X gate applied to qubit 1"); // |10⟩ (qubit 1 set)
        let ov = swap_test(&a, &b).expect("swap test on orthogonal 2-qubit states");
        assert!(ov.abs() < 1e-5, "overlap={ov}");
    }

    #[test]
    fn swap_test_mismatched_sizes_error() {
        let a = ket_zero(1);
        let b = ket_zero(2);
        assert!(swap_test(&a, &b).is_err());
    }

    #[test]
    fn hadamard_test_z_on_zero_is_plus_one() {
        let psi = ket_zero(1);
        let re = hadamard_test(&psi, &gate_z(), false)
            .expect("Hadamard test with Z gate on |0⟩, real part");
        assert!((re - 1.0).abs() < 1e-5, "re={re}");
    }

    #[test]
    fn hadamard_test_z_on_plus_is_zero() {
        let psi = ket_plus();
        let re = hadamard_test(&psi, &gate_z(), false)
            .expect("Hadamard test with Z gate on |+⟩, real part");
        assert!(re.abs() < 1e-5, "re={re}");
    }

    #[test]
    fn hadamard_test_x_on_zero_is_zero() {
        // ⟨0|X|0⟩ = 0.
        let psi = ket_zero(1);
        let re = hadamard_test(&psi, &gate_x(), false)
            .expect("Hadamard test with X gate on |0⟩, real part");
        assert!(re.abs() < 1e-5, "re={re}");
    }

    #[test]
    fn hadamard_test_s_on_plus_real_and_imag() {
        // ⟨+|S|+⟩ = (1 + i)/2.
        let psi = ket_plus();
        let re = hadamard_test(&psi, &gate_s(), false)
            .expect("Hadamard test with S gate on |+⟩, real part");
        let im = hadamard_test(&psi, &gate_s(), true)
            .expect("Hadamard test with S gate on |+⟩, imaginary part");
        assert!((re - 0.5).abs() < 1e-5, "re={re}");
        assert!((im - 0.5).abs() < 1e-5, "im={im}");
    }

    #[test]
    fn hadamard_test_requires_single_qubit() {
        let psi = ket_zero(2);
        assert!(hadamard_test(&psi, &gate_z(), false).is_err());
    }
}
