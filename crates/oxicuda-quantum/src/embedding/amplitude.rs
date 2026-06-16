use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

/// Amplitude embedding: load a real data vector as quantum amplitudes.
///
/// The data is L2-normalized and zero-padded to the next power of two.
/// Returns the state vector with `n_qubits = ceil(log2(data.len()))`.
pub fn amplitude_embedding(data: &[f32]) -> QuantumResult<StateVector> {
    if data.is_empty() {
        return Err(QuantumError::EmptyInput);
    }

    // Determine number of qubits needed
    let n_qubits = required_qubits(data.len())?;
    let dim = 1usize << n_qubits;

    // Build padded amplitude vector
    let mut amps = vec![Complex32::new(0.0, 0.0); dim];
    for (i, &v) in data.iter().enumerate() {
        amps[i] = Complex32::new(v, 0.0);
    }

    // Normalize
    let norm: f32 = amps.iter().map(|a| a.norm_sqr()).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return Err(QuantumError::NonNormalizedState { norm: 0.0 });
    }
    let inv = 1.0 / norm;
    for a in &mut amps {
        *a *= inv;
    }

    Ok(StateVector { amps, n_qubits })
}

fn required_qubits(n: usize) -> QuantumResult<usize> {
    if n == 0 {
        return Err(QuantumError::EmptyInput);
    }
    let nq = (usize::BITS - n.saturating_sub(1).leading_zeros()) as usize;
    let nq = nq.max(1);
    if nq > 30 {
        return Err(QuantumError::InvalidQubitCount { n: nq });
    }
    Ok(nq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amplitude_embedding_is_normalized() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let sv = amplitude_embedding(&data)
            .expect("data is non-empty with non-zero norm so amplitude embedding cannot fail");
        let norm = sv.norm_sq();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn amplitude_embedding_pads_to_power_of_two() {
        let data = [1.0_f32, 0.0, 0.0];
        let sv = amplitude_embedding(&data)
            .expect("data is non-empty with non-zero norm so amplitude embedding cannot fail");
        assert_eq!(sv.amps.len(), 4); // 2^2
    }

    #[test]
    fn amplitude_embedding_zero_vec_errors() {
        let data = [0.0_f32; 4];
        assert!(amplitude_embedding(&data).is_err());
    }
}
