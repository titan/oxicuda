use crate::error::QuantumResult;
use crate::gates::controlled::apply_cnot;
use crate::gates::hadamard::gate_h;
use crate::gates::parametric::{gate_rx, gate_rz};
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// QAOA circuit for the MaxCut problem on an unweighted graph.
///
/// Architecture: alternating cost (RZZ) and mixer (RX) layers for `p` rounds.
#[derive(Debug, Clone)]
pub struct QaoaCircuit {
    pub n_qubits: usize,
    pub p: usize,
    pub gammas: Vec<f32>,
    pub betas: Vec<f32>,
}

impl QaoaCircuit {
    /// Construct with given layer count and variational parameters.
    pub fn new(
        n_qubits: usize,
        p: usize,
        gammas: Vec<f32>,
        betas: Vec<f32>,
    ) -> QuantumResult<Self> {
        use crate::error::QuantumError;
        if gammas.len() != p || betas.len() != p {
            return Err(QuantumError::DimensionMismatch {
                expected: p,
                got: gammas.len().min(betas.len()),
            });
        }
        Ok(Self {
            n_qubits,
            p,
            gammas,
            betas,
        })
    }

    /// Apply RZZ(2γ) gate on qubits (i, j): exp(-i*γ*Z_i⊗Z_j).
    fn apply_rzz(sv: &mut StateVector, i: usize, j: usize, angle: f32) -> QuantumResult<()> {
        apply_cnot(sv, i, j)?;
        apply_1q_inplace(sv, j, &gate_rz(2.0 * angle))?;
        apply_cnot(sv, i, j)
    }

    /// Run the QAOA circuit starting from |+⟩^n and return the final state.
    pub fn run(&self, graph: &[(usize, usize)]) -> QuantumResult<StateVector> {
        let mut sv = StateVector::new_zero_state(self.n_qubits)?;
        let h = gate_h();

        // Initialize |+⟩^n = H^⊗n |0⟩^n
        for q in 0..self.n_qubits {
            apply_1q_inplace(&mut sv, q, &h)?;
        }

        for layer in 0..self.p {
            let gamma = self.gammas[layer];
            let beta = self.betas[layer];

            // Cost Hamiltonian: apply exp(-i*γ*Z_i⊗Z_j) for each edge
            for &(i, j) in graph {
                Self::apply_rzz(&mut sv, i, j, gamma)?;
            }

            // Mixer Hamiltonian: apply exp(-i*β*X_q) = Rx(2β) on each qubit
            for q in 0..self.n_qubits {
                apply_1q_inplace(&mut sv, q, &gate_rx(2.0 * beta))?;
            }
        }

        Ok(sv)
    }

    /// Evaluate the expected MaxCut cost C = Σ_{(i,j)∈E} ½(1 - ⟨Z_i Z_j⟩).
    pub fn energy(&self, sv: &StateVector, graph: &[(usize, usize)]) -> f32 {
        graph
            .iter()
            .map(|&(i, j)| {
                let mask_i = 1usize << i;
                let mask_j = 1usize << j;
                let zizj: f32 = sv
                    .amps
                    .iter()
                    .enumerate()
                    .map(|(idx, a)| {
                        let si = if (idx & mask_i) != 0 {
                            -1.0_f32
                        } else {
                            1.0_f32
                        };
                        let sj = if (idx & mask_j) != 0 {
                            -1.0_f32
                        } else {
                            1.0_f32
                        };
                        si * sj * a.norm_sqr()
                    })
                    .sum();
                0.5 * (1.0 - zizj)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qaoa_runs_without_error() {
        let circuit = QaoaCircuit::new(3, 1, vec![0.3], vec![0.5]).unwrap();
        let graph = vec![(0, 1), (1, 2)];
        let sv = circuit.run(&graph).unwrap();
        let norm = sv.norm_sq();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    #[test]
    fn qaoa_energy_non_negative() {
        let circuit = QaoaCircuit::new(3, 1, vec![0.3], vec![0.5]).unwrap();
        let graph = vec![(0, 1), (1, 2)];
        let sv = circuit.run(&graph).unwrap();
        let e = circuit.energy(&sv, &graph);
        assert!(e >= -1e-4, "energy={e}");
    }
}
