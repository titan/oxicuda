use crate::error::{QuantumError, QuantumResult};
use crate::handle::LcgRng;
use crate::pauli::expval::expectation_value;
use crate::pauli::hamiltonian::Hamiltonian;
use crate::statevec::state::StateVector;
use crate::vqe::ansatz::HardwareEfficientAnsatz;

/// VQE optimizer using gradient descent with the parameter-shift rule.
#[derive(Debug, Clone)]
pub struct VqeOptimizer {
    pub ansatz: HardwareEfficientAnsatz,
    pub ham: Hamiltonian,
    pub params: Vec<f32>,
}

impl VqeOptimizer {
    /// Construct the optimizer, initializing parameters with small random perturbations.
    pub fn new(ansatz: HardwareEfficientAnsatz, ham: Hamiltonian, rng: &mut LcgRng) -> Self {
        let n = ansatz.n_params();
        let params = (0..n).map(|_| rng.next_normal() * 0.1).collect();
        Self {
            ansatz,
            ham,
            params,
        }
    }

    /// Evaluate the energy ⟨ψ(params)|H|ψ(params)⟩.
    pub fn energy(&self, params: &[f32]) -> QuantumResult<f32> {
        let circ = self.ansatz.build_circuit(params)?;
        let mut rng = LcgRng::new(0);
        let sv = circ.exec_on_state(
            &StateVector::new_zero_state(self.ansatz.n_qubits)?,
            &mut rng,
        )?;
        expectation_value(&sv, &self.ham)
    }

    /// Compute gradient via the parameter-shift rule: ∂E/∂θ_k = ½(E(θ_k+π/2) - E(θ_k-π/2)).
    pub fn gradient(&self, params: &[f32]) -> QuantumResult<Vec<f32>> {
        let shift = std::f32::consts::FRAC_PI_2;
        let n = params.len();
        let mut grad = vec![0.0_f32; n];

        for k in 0..n {
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[k] += shift;
            p_minus[k] -= shift;
            let e_plus = self.energy(&p_plus)?;
            let e_minus = self.energy(&p_minus)?;
            grad[k] = 0.5 * (e_plus - e_minus);
        }

        Ok(grad)
    }

    /// Gradient-descent optimization loop.
    ///
    /// Returns `(final_energy, final_params)`.
    pub fn optimize(&mut self, max_iter: usize, lr: f32) -> QuantumResult<(f32, Vec<f32>)> {
        for iter in 0..max_iter {
            let grad = self.gradient(&self.params)?;
            for (p, g) in self.params.iter_mut().zip(grad.iter()) {
                *p -= lr * g;
            }
            let e = self.energy(&self.params)?;
            if !e.is_finite() {
                return Err(QuantumError::OptimizationDiverged { iter });
            }
        }
        let e = self.energy(&self.params)?;
        Ok((e, self.params.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::pauli_string::PauliOp;

    #[test]
    fn vqe_energy_is_finite() {
        let ans = HardwareEfficientAnsatz::new(2, 1);
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z, PauliOp::Z]);
        let mut rng = LcgRng::new(42);
        let opt = VqeOptimizer::new(ans, ham, &mut rng);
        let e = opt.energy(&opt.params.clone())
            .expect("energy evaluation on a freshly constructed optimizer with valid Hamiltonian cannot fail");
        assert!(e.is_finite(), "energy={e}");
    }
}
