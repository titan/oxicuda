use crate::error::QuantumResult;
use crate::pauli::hamiltonian::Hamiltonian;
use crate::pauli::pauli_string::{PauliOp, PauliString};
use crate::statevec::state::StateVector;

/// Compute ⟨ψ|H|ψ⟩ for a Hamiltonian H = Σ_k c_k P_k.
///
/// For each Pauli term P_k: rotates to the Z-eigenbasis, sums parity-weighted probabilities,
/// rotates back. This avoids forming the full 2^n × 2^n matrix.
pub fn expectation_value(sv: &StateVector, ham: &Hamiltonian) -> QuantumResult<f32> {
    let mut total = 0.0_f32;

    for (coeff, ops) in &ham.terms {
        if ops.is_empty() {
            total += coeff;
            continue;
        }

        let ps = PauliString::new(1.0, ops.clone());
        // ⟨P⟩ = ⟨ψ|P|ψ⟩ = real part of inner product ⟨ψ|P|ψ⟩
        let p_psi = ps.apply_to_state(sv)?;
        let ip = sv.inner_product(&p_psi)?;
        total += coeff * ip.re;
    }

    Ok(total)
}

/// Compute the expectation value of a single Pauli-Z string directly from probabilities.
///
/// This is an optimized path when all non-I operators are Z: no basis rotation needed.
/// ⟨Z⊗…⊗Z⟩ = Σ_i (-1)^{popcount(i & zmask)} |ψ_i|²
pub fn expval_z_string(sv: &StateVector, ops: &[PauliOp]) -> QuantumResult<f32> {
    let zmask: usize = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| **op == PauliOp::Z)
        .fold(0usize, |acc, (q, _)| acc | (1 << q));

    Ok(sv
        .amps
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let parity = (i & zmask).count_ones() & 1;
            let sign = if parity == 0 { 1.0_f32 } else { -1.0_f32 };
            sign * a.norm_sqr()
        })
        .sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::hamiltonian::Hamiltonian;
    use crate::statevec::state::StateVector;

    #[test]
    fn z_expval_of_zero_state_is_plus_one() {
        let sv = StateVector::new_zero_state(1)
            .expect("n_qubits=1 is a valid qubit count so zero-state construction cannot fail");
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z]);
        let ev = expectation_value(&sv, &ham)
            .expect("single-qubit Z expectation value on a normalized state is always computable");
        assert!((ev - 1.0).abs() < 1e-5, "ev={ev}");
    }
}
