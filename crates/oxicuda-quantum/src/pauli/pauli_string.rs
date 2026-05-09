use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

/// Single-qubit Pauli operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliOp {
    I,
    X,
    Y,
    Z,
}

impl PauliOp {
    /// Parse from a single character 'I', 'X', 'Y', or 'Z'.
    pub fn from_char(c: char) -> QuantumResult<Self> {
        match c {
            'I' => Ok(Self::I),
            'X' => Ok(Self::X),
            'Y' => Ok(Self::Y),
            'Z' => Ok(Self::Z),
            _ => Err(QuantumError::InvalidPauliOp { op: c.to_string() }),
        }
    }
}

/// A weighted tensor product of single-qubit Pauli operators.
///
/// `ops[i]` is the operator on qubit `i`. Length must equal `n_qubits`.
#[derive(Debug, Clone)]
pub struct PauliString {
    pub weight: f32,
    pub ops: Vec<PauliOp>,
}

impl PauliString {
    #[must_use]
    pub fn new(weight: f32, ops: Vec<PauliOp>) -> Self {
        Self { weight, ops }
    }

    /// Apply this Pauli string as an operator to the state vector, returning the new state.
    ///
    /// Computes P|ψ⟩ where P = weight * ⊗_i ops\[i\].
    pub fn apply_to_state(&self, sv: &StateVector) -> QuantumResult<StateVector> {
        if self.ops.len() != sv.n_qubits {
            return Err(QuantumError::DimensionMismatch {
                expected: sv.n_qubits,
                got: self.ops.len(),
            });
        }

        let dim = sv.amps.len();
        let mut out = vec![Complex32::new(0.0, 0.0); dim];

        for (idx, amp) in sv.amps.iter().enumerate() {
            let mut new_idx = idx;
            let mut phase = Complex32::new(self.weight, 0.0);

            for (q, op) in self.ops.iter().enumerate() {
                let bit = (idx >> q) & 1;
                match op {
                    PauliOp::I => {}
                    PauliOp::X => {
                        // X flips the bit
                        new_idx ^= 1 << q;
                    }
                    PauliOp::Y => {
                        // Y flips the bit and multiplies by i or -i
                        new_idx ^= 1 << q;
                        if bit == 0 {
                            // Y|0⟩ = i|1⟩
                            phase *= Complex32::new(0.0, 1.0);
                        } else {
                            // Y|1⟩ = -i|0⟩
                            phase *= Complex32::new(0.0, -1.0);
                        }
                    }
                    PauliOp::Z => {
                        // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
                        if bit == 1 {
                            phase *= Complex32::new(-1.0, 0.0);
                        }
                    }
                }
            }

            out[new_idx] += phase * amp;
        }

        Ok(StateVector {
            amps: out,
            n_qubits: sv.n_qubits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pauli_op_from_char() {
        assert_eq!(PauliOp::from_char('X').unwrap(), PauliOp::X);
        assert!(PauliOp::from_char('Q').is_err());
    }

    #[test]
    fn z_on_zero_state_no_change() {
        let sv = StateVector::new_zero_state(1).unwrap();
        let ps = PauliString::new(1.0, vec![PauliOp::Z]);
        let out = ps.apply_to_state(&sv).unwrap();
        // Z|0⟩ = |0⟩
        assert!((out.amps[0].re - 1.0).abs() < 1e-6);
    }
}
