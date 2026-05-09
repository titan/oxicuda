use crate::error::{QuantumError, QuantumResult};
use crate::gates::controlled::apply_cnot;
use crate::gates::hadamard::gate_h;
use crate::gates::parametric::{gate_phase, gate_rz};
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Havlíček ZZ feature map.
///
/// Circuit structure for `depth` repetitions:
///   1. H on all qubits
///   2. Rz(2·x\[i\]) on each qubit i
///   3. For each pair (i,j): CNOT(i,j) · Rz(2·x\[i\]·x\[j\]) · CNOT(i,j)
pub fn zz_feature_map(data: &[f32], depth: usize) -> QuantumResult<StateVector> {
    if data.is_empty() {
        return Err(QuantumError::EmptyInput);
    }
    let n = data.len();
    let mut sv = StateVector::new_zero_state(n)?;
    let h = gate_h();

    for _d in 0..depth {
        // H on all qubits
        for q in 0..n {
            apply_1q_inplace(&mut sv, q, &h)?;
        }

        // Single-qubit phase rotations Rz(2*x[i])
        for (i, &xi) in data.iter().enumerate() {
            apply_1q_inplace(&mut sv, i, &gate_rz(2.0 * xi))?;
        }

        // Two-qubit ZZ interactions
        for i in 0..n {
            for j in (i + 1)..n {
                let angle = 2.0 * data[i] * data[j];
                apply_cnot(&mut sv, i, j)?;
                apply_1q_inplace(&mut sv, j, &gate_rz(2.0 * angle))?;
                apply_cnot(&mut sv, i, j)?;
            }
        }
    }

    Ok(sv)
}

// Keep gate_phase in scope for potential single-qubit phase gate usage
fn _use_phase() -> [[num_complex::Complex<f32>; 2]; 2] {
    gate_phase(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zz_feature_map_preserves_norm() {
        let data = [0.5_f32, 1.0, -0.3];
        let sv = zz_feature_map(&data, 2).unwrap();
        let norm = sv.norm_sq();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    #[test]
    fn zz_feature_map_empty_errors() {
        assert!(zz_feature_map(&[], 1).is_err());
    }
}
