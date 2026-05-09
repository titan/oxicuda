use crate::error::QuantumResult;
use crate::gates::parametric::gate_ry;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Angle embedding: encode `data\[i\]` as RY(data\[i\]) on qubit i of |0⟩^n.
///
/// The number of qubits equals `data.len()`.
pub fn angle_embedding(data: &[f32]) -> QuantumResult<StateVector> {
    use crate::error::QuantumError;
    if data.is_empty() {
        return Err(QuantumError::EmptyInput);
    }

    let n = data.len();
    let mut sv = StateVector::new_zero_state(n)?;

    for (q, &angle) in data.iter().enumerate() {
        apply_1q_inplace(&mut sv, q, &gate_ry(angle))?;
    }

    Ok(sv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_embedding_norm_preserved() {
        let data = [0.5_f32, 1.2, 0.3];
        let sv = angle_embedding(&data).unwrap();
        let norm = sv.norm_sq();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn angle_embedding_empty_errors() {
        assert!(angle_embedding(&[]).is_err());
    }
}
