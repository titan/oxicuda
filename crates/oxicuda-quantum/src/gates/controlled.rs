use crate::error::QuantumResult;
use crate::gates::hadamard::{gate_t, gate_tdg};
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
    // Canonical Nielsen & Chuang Toffoli (CCX) decomposition into H, T, T†, CNOT.
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
    apply_cnot(sv, c0, c1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cnot_creates_bell_state() {
        use crate::gates::hadamard::gate_h;
        use crate::statevec::apply_1q::apply_1q_inplace;

        let mut sv = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("H gate on qubit 0 should succeed");
        apply_cnot(&mut sv, 0, 1).expect("CNOT on qubits 0 to 1 should succeed");

        // Bell state: |00⟩ + |11⟩ / √2
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((sv.amps[0].re - inv_sqrt2).abs() < 1e-5);
        assert!((sv.amps[3].re - inv_sqrt2).abs() < 1e-5);
        assert!(sv.amps[1].norm() < 1e-5);
        assert!(sv.amps[2].norm() < 1e-5);
    }

    #[test]
    fn ccx_is_true_toffoli() {
        use crate::statevec::apply_1q::apply_1q_inplace;
        // LSB ordering: qubit q ↔ bit (1<<q). ccx(c0=0,c1=1,tgt=2) flips bit-2
        // (value 4) iff bit-0 and bit-1 are both set. The result amplitude must be
        // EXACTLY +1 — the old S†·Toffoli·S = CC(−Y) miswiring would give ∓i.
        // q0=1,q1=1,q2=0 (index 3) → flip q2 → index 7.
        let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_x()).expect("X gate on qubit 0 should succeed");
        apply_1q_inplace(&mut sv, 1, &gate_x()).expect("X gate on qubit 1 should succeed");
        apply_ccx(&mut sv, 0, 1, 2).expect("CCX gate with both controls set should succeed");
        assert!((sv.amps[7].re - 1.0).abs() < 1e-5, "re={}", sv.amps[7].re);
        assert!(sv.amps[7].im.abs() < 1e-5, "im={}", sv.amps[7].im);
        for (i, a) in sv.amps.iter().enumerate() {
            if i != 7 {
                assert!(a.norm() < 1e-5, "spurious amp[{i}]={a:?}");
            }
        }

        // q0=q1=q2=1 (index 7) → flip q2 → index 3, amplitude +1.
        let mut sv =
            StateVector::new_zero_state(3).expect("valid 3-qubit zero state for all-ones case");
        for q in 0..3 {
            apply_1q_inplace(&mut sv, q, &gate_x())
                .expect("X gate on each qubit for all-ones preparation should succeed");
        }
        apply_ccx(&mut sv, 0, 1, 2).expect("CCX on all-ones state should succeed");
        assert!((sv.amps[3].re - 1.0).abs() < 1e-5, "re={}", sv.amps[3].re);
        assert!(sv.amps[3].im.abs() < 1e-5, "im={}", sv.amps[3].im);

        // Only one control set (q0=1, index 1) → target unchanged.
        let mut sv = StateVector::new_zero_state(3)
            .expect("valid 3-qubit zero state for single-control case");
        apply_1q_inplace(&mut sv, 0, &gate_x())
            .expect("X gate on qubit 0 for single-control setup should succeed");
        apply_ccx(&mut sv, 0, 1, 2).expect("CCX with single control set should succeed");
        assert!((sv.amps[1].re - 1.0).abs() < 1e-5);
        assert!(sv.amps[1].im.abs() < 1e-5);

        // No controls set → |000⟩ unchanged.
        let mut sv =
            StateVector::new_zero_state(3).expect("valid 3-qubit zero state for no-control case");
        apply_ccx(&mut sv, 0, 1, 2).expect("CCX with no controls set should succeed");
        assert!((sv.amps[0].re - 1.0).abs() < 1e-5);
    }
}
