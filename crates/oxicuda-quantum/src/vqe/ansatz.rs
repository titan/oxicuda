use crate::circuit::circuit::{GateOp, QuantumCircuit};
use crate::error::{QuantumError, QuantumResult};

/// Hardware-efficient ansatz: layers of RY rotations interleaved with CNOT entanglers.
///
/// Parameter layout: `params[layer * n_qubits + qubit]` = RY angle for that qubit/layer.
#[derive(Debug, Clone)]
pub struct HardwareEfficientAnsatz {
    pub n_qubits: usize,
    pub depth: usize,
}

impl HardwareEfficientAnsatz {
    #[must_use]
    pub fn new(n_qubits: usize, depth: usize) -> Self {
        Self { n_qubits, depth }
    }

    /// Total number of variational parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        (self.depth + 1) * self.n_qubits
    }

    /// Build the parametric circuit from the given parameter vector.
    pub fn build_circuit(&self, params: &[f32]) -> QuantumResult<QuantumCircuit> {
        if params.len() != self.n_params() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_params(),
                got: params.len(),
            });
        }
        let mut circ = QuantumCircuit::new(self.n_qubits);

        for layer in 0..=self.depth {
            // RY layer
            for q in 0..self.n_qubits {
                let theta = params[layer * self.n_qubits + q];
                circ.add_gate(GateOp::Ry(theta));
            }
            // CNOT entanglement layer (not on last layer to keep expressibility)
            if layer < self.depth {
                for q in 0..(self.n_qubits - 1) {
                    circ.add_gate(GateOp::Cnot {
                        ctrl: q,
                        tgt: q + 1,
                    });
                }
            }
        }

        Ok(circ)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_params_correct() {
        let ans = HardwareEfficientAnsatz::new(4, 2);
        // (depth+1) * n_qubits = 3 * 4 = 12
        assert_eq!(ans.n_params(), 12);
    }

    #[test]
    fn build_circuit_wrong_params_errors() {
        let ans = HardwareEfficientAnsatz::new(2, 1);
        let params = vec![0.0_f32; 5]; // wrong size
        assert!(ans.build_circuit(&params).is_err());
    }
}
