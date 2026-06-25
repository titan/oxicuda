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
        let mut sv = StateVector::new_zero_state(2).expect(
            "n_qubits=2 is always a valid qubit count, so zero-state construction cannot fail",
        );
        ts.step_2nd(&mut sv, 0.1)
            .expect("Hamiltonian has a valid 2-qubit Pauli term and dt=0.1 is finite, so the second-order Trotter step cannot fail");
        let norm = sv.norm_sq();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }
}

/// Numerical-accuracy cross-checks of Trotter orders against an exact dense
/// matrix exponential `expm(-iHt)` for the transverse-field XX-Ising model.
#[cfg(test)]
mod accuracy_tests {
    use super::*;
    use num_complex::Complex;

    type C = Complex<f32>;

    /// Dense matrix of a unit-weight Pauli string acting as an operator.
    fn pauli_matrix(ops: &[PauliOp], n_qubits: usize) -> Vec<C> {
        use crate::pauli::pauli_string::PauliString;
        let dim = 1usize << n_qubits;
        let mut mat = vec![C::new(0.0, 0.0); dim * dim];
        let ps = PauliString::new(1.0, ops.to_vec());
        for col in 0..dim {
            let mut e = vec![C::new(0.0, 0.0); dim];
            e[col] = C::new(1.0, 0.0);
            let sv = StateVector { amps: e, n_qubits };
            let out = ps
                .apply_to_state(&sv)
                .expect("unit Pauli string on a basis vector of matching width cannot fail");
            for row in 0..dim {
                mat[row * dim + col] = out.amps[row];
            }
        }
        mat
    }

    /// Build the dense Hamiltonian Σ coeff·P.
    fn dense_hamiltonian(ham: &Hamiltonian, n_qubits: usize) -> Vec<C> {
        let dim = 1usize << n_qubits;
        let mut h = vec![C::new(0.0, 0.0); dim * dim];
        for (coeff, ops) in &ham.terms {
            let p = pauli_matrix(ops, n_qubits);
            for (a, b) in h.iter_mut().zip(p.iter()) {
                *a += *coeff * b;
            }
        }
        h
    }

    fn mat_mul(a: &[C], b: &[C], dim: usize) -> Vec<C> {
        let mut c = vec![C::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut acc = C::new(0.0, 0.0);
                for k in 0..dim {
                    acc += a[i * dim + k] * b[k * dim + j];
                }
                c[i * dim + j] = acc;
            }
        }
        c
    }

    /// Exact `expm(M)` via scaling-and-squaring with a Taylor series.
    fn expm(m: &[C], dim: usize) -> Vec<C> {
        // Scale M by 2^-s so ‖M/2^s‖ is small, Taylor-expand, then square s times.
        let max_abs = m.iter().map(|z| z.norm()).fold(0.0_f32, f32::max);
        let s = (max_abs.log2().ceil().max(0.0)) as u32 + 4;
        let scale = 0.5_f32.powi(s as i32);
        let ms: Vec<C> = m.iter().map(|z| z * scale).collect();

        // Taylor: I + ms + ms²/2! + … (24 terms is ample for a well-scaled arg).
        let mut result = vec![C::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            result[i * dim + i] = C::new(1.0, 0.0);
        }
        let mut term = result.clone(); // current power / k!
        for k in 1..24u32 {
            term = mat_mul(&term, &ms, dim);
            let inv = 1.0 / k as f32;
            for t in term.iter_mut() {
                *t *= inv;
            }
            for (r, t) in result.iter_mut().zip(term.iter()) {
                *r += t;
            }
        }
        // Square s times.
        for _ in 0..s {
            result = mat_mul(&result, &result, dim);
        }
        result
    }

    fn mat_vec(m: &[C], v: &[C], dim: usize) -> Vec<C> {
        let mut out = vec![C::new(0.0, 0.0); dim];
        for i in 0..dim {
            let mut acc = C::new(0.0, 0.0);
            for j in 0..dim {
                acc += m[i * dim + j] * v[j];
            }
            out[i] = acc;
        }
        out
    }

    /// L2 distance between two amplitude vectors.
    fn l2(a: &[C], b: &[C]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm_sqr())
            .sum::<f32>()
            .sqrt()
    }

    /// Transverse-field XX-Ising on 2 qubits: H = J·XX + h·(ZI + IZ).
    fn xx_ising() -> Hamiltonian {
        let mut ham = Hamiltonian::new();
        ham.add_term(0.8, vec![PauliOp::X, PauliOp::X]);
        ham.add_term(0.5, vec![PauliOp::Z, PauliOp::I]);
        ham.add_term(0.5, vec![PauliOp::I, PauliOp::Z]);
        ham
    }

    /// Run `n_steps` Trotter steps of the given order to evolve |+0⟩ for total
    /// time `t`, returning the resulting amplitudes.
    fn trotter_evolve(order: u8, n_steps: usize, t: f32) -> Vec<C> {
        let ts = TrotterStep::new(xx_ising(), order);
        let mut sv = StateVector::new_zero_state(2).expect("2q");
        // Prepare a non-trivial initial state |+0⟩.
        apply_1q_inplace(&mut sv, 0, &crate::gates::hadamard::gate_h()).expect("h");
        let dt = t / n_steps as f32;
        for _ in 0..n_steps {
            match order {
                1 => ts.step_1st(&mut sv, dt).expect("1st"),
                2 => ts.step_2nd(&mut sv, dt).expect("2nd"),
                _ => ts.step_4th(&mut sv, dt).expect("4th"),
            }
        }
        sv.amps
    }

    /// Exact reference evolution via expm(-iHt).
    fn exact_evolve(t: f32) -> Vec<C> {
        let h = dense_hamiltonian(&xx_ising(), 2);
        // -i t H.
        let arg: Vec<C> = h.iter().map(|z| C::new(0.0, -t) * z).collect();
        let u = expm(&arg, 4);
        let mut sv = StateVector::new_zero_state(2).expect("2q");
        apply_1q_inplace(&mut sv, 0, &crate::gates::hadamard::gate_h()).expect("h");
        mat_vec(&u, &sv.amps, 4)
    }

    #[test]
    fn expm_of_zero_is_identity() {
        let zero = vec![C::new(0.0, 0.0); 4];
        let e = expm(&zero, 2);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((e[i * 2 + j].re - expected).abs() < 1e-6);
                assert!(e[i * 2 + j].im.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn trotter_error_decreases_with_step_count() {
        let t = 1.0_f32;
        let exact = exact_evolve(t);
        let err_coarse = l2(&trotter_evolve(1, 4, t), &exact);
        let err_fine = l2(&trotter_evolve(1, 32, t), &exact);
        assert!(
            err_fine < err_coarse,
            "finer Trotter should be more accurate: coarse={err_coarse}, fine={err_fine}"
        );
    }

    #[test]
    fn higher_order_trotter_is_more_accurate() {
        // At a fixed, deliberately coarse step count, error ordering 4th ≤ 2nd ≤ 1st.
        let t = 1.0_f32;
        let steps = 6usize;
        let exact = exact_evolve(t);
        let e1 = l2(&trotter_evolve(1, steps, t), &exact);
        let e2 = l2(&trotter_evolve(2, steps, t), &exact);
        let e4 = l2(&trotter_evolve(4, steps, t), &exact);
        assert!(e2 <= e1 + 1e-4, "2nd ({e2}) should beat 1st ({e1})");
        assert!(e4 <= e2 + 1e-4, "4th ({e4}) should beat 2nd ({e2})");
        // 4th order should be very accurate even at 6 steps.
        assert!(e4 < 1e-2, "4th-order error too large: {e4}");
    }
}
