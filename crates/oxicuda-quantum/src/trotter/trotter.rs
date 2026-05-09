use crate::error::QuantumResult;
use crate::gates::controlled::apply_cnot;
use crate::gates::parametric::{gate_rx, gate_rz};
use crate::pauli::hamiltonian::Hamiltonian;
use crate::pauli::pauli_string::PauliOp;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Trotter-Suzuki integrator for time evolution under a Pauli Hamiltonian.
#[derive(Debug, Clone)]
pub struct TrotterStep {
    pub ham: Hamiltonian,
    pub order: u8,
}

impl TrotterStep {
    #[must_use]
    pub fn new(ham: Hamiltonian, order: u8) -> Self {
        Self { ham, order }
    }

    /// Apply exp(-i·c·dt·P_k)|ψ⟩ for a single Pauli term, in place.
    ///
    /// Strategy: rotate to Z-basis, apply Rz, rotate back.
    fn apply_pauli_exp(
        sv: &mut StateVector,
        coeff: f32,
        ops: &[PauliOp],
        dt: f32,
    ) -> QuantumResult<()> {
        let n = sv.n_qubits;

        // Basis rotation: X→ H, Y→ HS†
        for (q, op) in ops.iter().enumerate() {
            match op {
                PauliOp::X => {
                    apply_1q_inplace(sv, q, &crate::gates::hadamard::gate_h())?;
                }
                PauliOp::Y => {
                    apply_1q_inplace(sv, q, &crate::gates::hadamard::gate_sdg())?;
                    apply_1q_inplace(sv, q, &crate::gates::hadamard::gate_h())?;
                }
                PauliOp::I | PauliOp::Z => {}
            }
        }

        // Collect active qubit indices (non-I ops)
        let active: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(_, op)| **op != PauliOp::I)
            .map(|(q, _)| q)
            .collect();

        // CNOT cascade to accumulate parity into last active qubit
        if active.len() >= 2 {
            for k in 0..(active.len() - 1) {
                apply_cnot(sv, active[k], active[k + 1])?;
            }
        }

        // Apply Rz(2*coeff*dt) to the last active qubit (parity register)
        if let Some(&last) = active.last() {
            apply_1q_inplace(sv, last, &gate_rz(2.0 * coeff * dt))?;
        } else {
            // All-I term: global phase only (observable invariant)
        }

        // Undo CNOT cascade
        if active.len() >= 2 {
            for k in (0..(active.len() - 1)).rev() {
                apply_cnot(sv, active[k], active[k + 1])?;
            }
        }

        // Undo basis rotation
        for (q, op) in ops.iter().enumerate() {
            match op {
                PauliOp::X => {
                    apply_1q_inplace(sv, q, &crate::gates::hadamard::gate_h())?;
                }
                PauliOp::Y => {
                    apply_1q_inplace(sv, q, &crate::gates::hadamard::gate_h())?;
                    apply_1q_inplace(sv, q, &crate::gates::hadamard::gate_s())?;
                }
                PauliOp::I | PauliOp::Z => {}
            }
        }

        let _ = n; // suppress unused warning if ham has 0 active qubits
        Ok(())
    }

    /// First-order Suzuki-Trotter step: e^{-iHdt} ≈ ∏_k e^{-ic_k·dt·P_k}.
    pub fn step_1st(&self, sv: &mut StateVector, dt: f32) -> QuantumResult<()> {
        for (coeff, ops) in &self.ham.terms {
            Self::apply_pauli_exp(sv, *coeff, ops, dt)?;
        }
        Ok(())
    }

    /// Second-order Suzuki-Trotter step (symmetric split).
    pub fn step_2nd(&self, sv: &mut StateVector, dt: f32) -> QuantumResult<()> {
        let half = dt * 0.5;
        for (coeff, ops) in &self.ham.terms {
            Self::apply_pauli_exp(sv, *coeff, ops, half)?;
        }
        for (coeff, ops) in self.ham.terms.iter().rev() {
            Self::apply_pauli_exp(sv, *coeff, ops, half)?;
        }
        Ok(())
    }

    /// Fourth-order Yoshida Trotter step.
    ///
    /// Coefficients: s = 1/(2 - 2^{1/3}), w1 = s, w0 = 1 - 2s.
    pub fn step_4th(&self, sv: &mut StateVector, dt: f32) -> QuantumResult<()> {
        let s = 1.0 / (2.0 - 2.0_f32.powf(1.0 / 3.0));
        let w0 = 1.0 - 2.0 * s;
        let w1 = s;
        // Yoshida composition: S2(w1*dt) · S2(w0*dt) · S2(w1*dt)
        self.step_2nd(sv, w1 * dt)?;
        self.step_2nd(sv, w0 * dt)?;
        self.step_2nd(sv, w1 * dt)
    }
}

// Keep rx in scope for potential future terms (suppress dead_code)
fn _use_rx() -> [[num_complex::Complex<f32>; 2]; 2] {
    gate_rx(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::hamiltonian::Hamiltonian;
    use crate::pauli::pauli_string::PauliOp;

    #[test]
    fn trotter_step_preserves_norm() {
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z, PauliOp::I]);
        let ts = TrotterStep::new(ham, 2);
        let mut sv = StateVector::new_zero_state(2).unwrap();
        ts.step_2nd(&mut sv, 0.1).unwrap();
        let norm = sv.norm_sq();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }
}
