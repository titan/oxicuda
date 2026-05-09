use crate::pauli::pauli_string::PauliOp;

/// A Hamiltonian as a sum of weighted Pauli string terms.
#[derive(Debug, Clone, Default)]
pub struct Hamiltonian {
    pub terms: Vec<(f32, Vec<PauliOp>)>,
}

impl Hamiltonian {
    #[must_use]
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Add a term `coeff * ⊗ops` to the Hamiltonian.
    pub fn add_term(&mut self, coeff: f32, ops: Vec<PauliOp>) {
        self.terms.push((coeff, ops));
    }

    /// Number of qubits inferred from the first term.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.terms.first().map(|(_, ops)| ops.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamiltonian_n_qubits() {
        let mut h = Hamiltonian::new();
        h.add_term(1.0, vec![PauliOp::Z, PauliOp::I]);
        assert_eq!(h.n_qubits(), 2);
    }
}
