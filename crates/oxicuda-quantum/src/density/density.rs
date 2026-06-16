use num_complex::Complex;

use crate::error::QuantumResult;
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

/// Density matrix ρ stored as a row-major dim×dim complex matrix.
#[derive(Debug, Clone)]
pub struct DensityMatrix {
    pub rho: Vec<Complex32>,
    pub dim: usize,
}

impl DensityMatrix {
    /// Construct ρ = |ψ⟩⟨ψ| from a pure state vector.
    #[must_use]
    pub fn from_pure_state(sv: &StateVector) -> Self {
        let dim = sv.amps.len();
        let mut rho = vec![Complex32::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                rho[i * dim + j] = sv.amps[i] * sv.amps[j].conj();
            }
        }
        Self { rho, dim }
    }

    /// Apply a unitary U: ρ' = U ρ U†.
    pub fn apply_unitary(&mut self, u: &[Complex32], dim: usize) -> QuantumResult<()> {
        use crate::error::QuantumError;
        if dim != self.dim {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dim,
                got: dim,
            });
        }
        // u_rho = U * ρ
        let mut u_rho = vec![Complex32::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    u_rho[i * dim + j] += u[i * dim + k] * self.rho[k * dim + j];
                }
            }
        }
        // result = u_rho * U†
        let mut result = vec![Complex32::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    // U†[j,k] = conj(U[k,j])
                    result[i * dim + j] += u_rho[i * dim + k] * u[j * dim + k].conj();
                }
            }
        }
        self.rho = result;
        Ok(())
    }

    /// Trace Tr(ρ) = Σ_i ρ\[i,i\].
    #[must_use]
    pub fn trace(&self) -> Complex32 {
        (0..self.dim)
            .map(|i| self.rho[i * self.dim + i])
            .fold(Complex32::new(0.0, 0.0), |acc, x| acc + x)
    }

    /// Tr(ρ²) used for purity calculation.
    #[must_use]
    pub fn trace_sq(&self) -> Complex32 {
        let dim = self.dim;
        let mut tr2 = Complex32::new(0.0, 0.0);
        for i in 0..dim {
            for k in 0..dim {
                // (ρ²)[i,i] = Σ_k ρ[i,k] * ρ[k,i]
                tr2 += self.rho[i * dim + k] * self.rho[k * dim + i];
            }
        }
        tr2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_state_trace_is_one() {
        let sv = StateVector::new_zero_state(2)
            .expect("n_qubits=2 is a valid qubit count so zero-state construction cannot fail");
        let dm = DensityMatrix::from_pure_state(&sv);
        let tr = dm.trace();
        assert!((tr.re - 1.0).abs() < 1e-6, "trace={}", tr.re);
    }

    #[test]
    fn pure_state_trace_sq_is_one() {
        let sv = StateVector::new_zero_state(1)
            .expect("n_qubits=1 is a valid qubit count so zero-state construction cannot fail");
        let dm = DensityMatrix::from_pure_state(&sv);
        let tr2 = dm.trace_sq();
        assert!((tr2.re - 1.0).abs() < 1e-5, "trace_sq={}", tr2.re);
    }
}
