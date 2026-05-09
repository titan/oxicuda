use crate::error::QuantumResult;
use crate::gates::hadamard::{gate_s, gate_sdg, gate_t, gate_tdg};
use crate::gates::pauli::gate_x;
use crate::statevec::apply_1q::{apply_1q_controlled, apply_1q_inplace};
use crate::statevec::state::StateVector;

/// CNOT (controlled-X) gate.
pub fn apply_cnot(sv: &mut StateVector, ctrl: usize, tgt: usize) -> QuantumResult<()> {
    apply_1q_controlled(sv, ctrl, tgt, &gate_x())
}

/// CZ (controlled-Z) gate.
pub fn apply_cz(sv: &mut StateVector, ctrl: usize, tgt: usize) -> QuantumResult<()> {
    use crate::gates::pauli::gate_z;
    apply_1q_controlled(sv, ctrl, tgt, &gate_z())
}

/// SWAP gate: swaps amplitude of qubits q0 and q1.
///
/// Implemented as CNOT(q0,q1) · CNOT(q1,q0) · CNOT(q0,q1).
pub fn apply_swap(sv: &mut StateVector, q0: usize, q1: usize) -> QuantumResult<()> {
    apply_cnot(sv, q0, q1)?;
    apply_cnot(sv, q1, q0)?;
    apply_cnot(sv, q0, q1)
}

/// Toffoli (CCX) gate: apply X to `tgt` when both `c0` and `c1` are |1⟩.
///
/// Decomposed via standard Hadamard + CNOT + T/Tdg circuit (7 T-gates).
pub fn apply_ccx(sv: &mut StateVector, c0: usize, c1: usize, tgt: usize) -> QuantumResult<()> {
    apply_1q_inplace(sv, tgt, &gate_s())?; // H on tgt
    apply_1q_inplace(sv, tgt, &crate::gates::hadamard::gate_h())?;
    apply_cnot(sv, c1, tgt)?;
    apply_1q_inplace(sv, tgt, &gate_tdg())?;
    apply_cnot(sv, c0, tgt)?;
    apply_1q_inplace(sv, tgt, &gate_t())?;
    apply_cnot(sv, c1, tgt)?;
    apply_1q_inplace(sv, tgt, &gate_tdg())?;
    apply_cnot(sv, c0, tgt)?;
    apply_1q_inplace(sv, c1, &gate_t())?;
    apply_1q_inplace(sv, tgt, &gate_t())?;
    apply_1q_inplace(sv, tgt, &crate::gates::hadamard::gate_h())?;
    apply_cnot(sv, c0, c1)?;
    apply_1q_inplace(sv, c0, &gate_t())?;
    apply_1q_inplace(sv, c1, &gate_tdg())?;
    apply_cnot(sv, c0, c1)?;
    apply_1q_inplace(sv, tgt, &gate_sdg())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cnot_creates_bell_state() {
        use crate::gates::hadamard::gate_h;
        use crate::statevec::apply_1q::apply_1q_inplace;

        let mut sv = StateVector::new_zero_state(2).unwrap();
        apply_1q_inplace(&mut sv, 0, &gate_h()).unwrap();
        apply_cnot(&mut sv, 0, 1).unwrap();

        // Bell state: |00⟩ + |11⟩ / √2
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((sv.amps[0].re - inv_sqrt2).abs() < 1e-5);
        assert!((sv.amps[3].re - inv_sqrt2).abs() < 1e-5);
        assert!(sv.amps[1].norm() < 1e-5);
        assert!(sv.amps[2].norm() < 1e-5);
    }
}
