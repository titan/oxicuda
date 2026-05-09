use crate::error::{QuantumError, QuantumResult};
use crate::gates::controlled::{apply_cnot, apply_cz, apply_swap};
use crate::gates::hadamard::{gate_h, gate_s, gate_t};
use crate::gates::parametric::{gate_rx, gate_ry, gate_rz};
use crate::gates::pauli::{gate_x, gate_y, gate_z};
use crate::handle::LcgRng;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// A single quantum gate operation in the circuit.
#[derive(Debug, Clone)]
pub enum GateOp {
    H,
    X,
    Y,
    Z,
    S,
    T,
    Rx(f32),
    Ry(f32),
    Rz(f32),
    Cnot { ctrl: usize, tgt: usize },
    Cz { ctrl: usize, tgt: usize },
    Swap { q0: usize, q1: usize },
    Measure { qubit: usize },
}

/// A sequence of gate operations acting on `n_qubits`.
///
/// Gates are stored in application order (first gate at index 0).
/// The circuit tracks which qubit each gate targets via internal qubit cursors.
///
/// Single-qubit gates (H, X, Y, Z, S, T, Rx, Ry, Rz) are applied to qubits
/// in round-robin order unless they carry an explicit qubit index. For simplicity
/// in this implementation, single-qubit gates carry an implicit cursor that advances
/// modulo n_qubits. Multi-qubit gates always carry explicit qubit indices.
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub n_qubits: usize,
    pub ops: Vec<(usize, GateOp)>,
    cursor: usize,
}

impl QuantumCircuit {
    #[must_use]
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            ops: Vec::new(),
            cursor: 0,
        }
    }

    /// Add a gate to the circuit. Single-qubit gates are assigned the next qubit
    /// in round-robin order; multi-qubit gates carry embedded qubit indices.
    pub fn add_gate(&mut self, op: GateOp) {
        let qubit = match &op {
            GateOp::Cnot { ctrl, .. } => *ctrl,
            GateOp::Cz { ctrl, .. } => *ctrl,
            GateOp::Swap { q0, .. } => *q0,
            GateOp::Measure { qubit } => *qubit,
            _ => {
                let q = self.cursor % self.n_qubits;
                self.cursor += 1;
                q
            }
        };
        self.ops.push((qubit, op));
    }

    /// Execute the circuit on a state vector, returning the post-execution state.
    pub fn exec_on_state(&self, sv: &StateVector, rng: &mut LcgRng) -> QuantumResult<StateVector> {
        if sv.n_qubits != self.n_qubits {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_qubits,
                got: sv.n_qubits,
            });
        }

        let mut state = sv.clone();

        for (qubit, op) in &self.ops {
            let q = *qubit;
            match op {
                GateOp::H => apply_1q_inplace(&mut state, q, &gate_h())?,
                GateOp::X => apply_1q_inplace(&mut state, q, &gate_x())?,
                GateOp::Y => apply_1q_inplace(&mut state, q, &gate_y())?,
                GateOp::Z => apply_1q_inplace(&mut state, q, &gate_z())?,
                GateOp::S => apply_1q_inplace(&mut state, q, &gate_s())?,
                GateOp::T => apply_1q_inplace(&mut state, q, &gate_t())?,
                GateOp::Rx(theta) => apply_1q_inplace(&mut state, q, &gate_rx(*theta))?,
                GateOp::Ry(theta) => apply_1q_inplace(&mut state, q, &gate_ry(*theta))?,
                GateOp::Rz(theta) => apply_1q_inplace(&mut state, q, &gate_rz(*theta))?,
                GateOp::Cnot { ctrl, tgt } => apply_cnot(&mut state, *ctrl, *tgt)?,
                GateOp::Cz { ctrl, tgt } => apply_cz(&mut state, *ctrl, *tgt)?,
                GateOp::Swap { q0, q1 } => apply_swap(&mut state, *q0, *q1)?,
                GateOp::Measure { qubit } => {
                    let (_, new_state) = state.sample_measure(*qubit, rng)?;
                    state = new_state;
                }
            }
        }

        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_h_then_h_returns_to_zero() {
        let mut circ = QuantumCircuit::new(1);
        circ.add_gate(GateOp::H);
        circ.add_gate(GateOp::H);
        let sv = StateVector::new_zero_state(1).unwrap();
        let mut rng = LcgRng::new(1);
        let out = circ.exec_on_state(&sv, &mut rng).unwrap();
        assert!(
            (out.amps[0].re - 1.0).abs() < 1e-5,
            "amp[0]={:?}",
            out.amps[0]
        );
        assert!(out.amps[1].norm() < 1e-5);
    }

    #[test]
    fn circuit_mismatch_errors() {
        let circ = QuantumCircuit::new(2);
        let sv = StateVector::new_zero_state(3).unwrap();
        let mut rng = LcgRng::new(0);
        assert!(circ.exec_on_state(&sv, &mut rng).is_err());
    }
}
