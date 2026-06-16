use num_complex::Complex;

use crate::density::density::DensityMatrix;
use crate::error::{QuantumError, QuantumResult};

type Complex32 = Complex<f32>;

/// Compute the reduced density matrix by tracing out all qubits not in `keep_qubits`.
///
/// `n_total` is the total number of qubits; `keep_qubits` is sorted in ascending order.
/// The output density matrix has dimension `2^|keep_qubits|`.
pub fn partial_trace(
    dm: &DensityMatrix,
    keep_qubits: &[usize],
    n_total: usize,
) -> QuantumResult<DensityMatrix> {
    let full_dim = 1usize << n_total;
    if dm.dim != full_dim {
        return Err(QuantumError::DimensionMismatch {
            expected: full_dim,
            got: dm.dim,
        });
    }

    for &q in keep_qubits {
        if q >= n_total {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: q,
                n_qubits: n_total,
            });
        }
    }

    let n_keep = keep_qubits.len();
    let reduced_dim = 1usize << n_keep;
    let mut rho_red = vec![Complex32::new(0.0, 0.0); reduced_dim * reduced_dim];

    // Build the set of traced-out (discarded) qubit indices
    let mut trace_qubits: Vec<usize> = (0..n_total).filter(|q| !keep_qubits.contains(q)).collect();
    trace_qubits.sort_unstable();
    let n_trace = trace_qubits.len();
    let trace_dim = 1usize << n_trace;

    // For each pair of keep-basis indices (i_keep, j_keep), sum over trace index
    for ri in 0..reduced_dim {
        for rj in 0..reduced_dim {
            let mut sum = Complex32::new(0.0, 0.0);

            for tk in 0..trace_dim {
                // Reconstruct full indices by interleaving keep and trace bits
                let full_i = interleave_bits(ri, tk, keep_qubits, &trace_qubits, n_total);
                let full_j = interleave_bits(rj, tk, keep_qubits, &trace_qubits, n_total);
                sum += dm.rho[full_i * full_dim + full_j];
            }

            rho_red[ri * reduced_dim + rj] = sum;
        }
    }

    Ok(DensityMatrix {
        rho: rho_red,
        dim: reduced_dim,
    })
}

/// Reconstruct a full-system basis index from keep-bits and trace-bits.
fn interleave_bits(
    keep_idx: usize,
    trace_idx: usize,
    keep_qubits: &[usize],
    trace_qubits: &[usize],
    n_total: usize,
) -> usize {
    let _ = n_total;
    let mut result = 0usize;
    for (k, &q) in keep_qubits.iter().enumerate() {
        if (keep_idx >> k) & 1 == 1 {
            result |= 1 << q;
        }
    }
    for (k, &q) in trace_qubits.iter().enumerate() {
        if (trace_idx >> k) & 1 == 1 {
            result |= 1 << q;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statevec::state::StateVector;

    #[test]
    fn partial_trace_2q_product_state() {
        // |0⟩ ⊗ |0⟩ → trace out qubit 1 → still |0⟩⟨0|
        let sv = StateVector::new_zero_state(2)
            .expect("n_qubits=2 is a valid qubit count so zero-state construction cannot fail");
        let dm = DensityMatrix::from_pure_state(&sv);
        let red = partial_trace(&dm, &[0], 2)
            .expect("keeping qubit 0 from a 2-qubit 4×4 density matrix is a valid operation");
        assert_eq!(red.dim, 2);
        assert!((red.rho[0].re - 1.0).abs() < 1e-5);
    }
}
