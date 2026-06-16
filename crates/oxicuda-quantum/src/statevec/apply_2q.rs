use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

/// Apply a 4×4 two-qubit gate in-place.
///
/// The gate matrix is indexed in the computational basis with q0 as the most-significant
/// of the two qubits in the 2-qubit subspace: |q0 q1⟩.
pub fn apply_2q_inplace(
    sv: &mut StateVector,
    q0: usize,
    q1: usize,
    gate: &[[Complex32; 4]; 4],
) -> QuantumResult<()> {
    if q0 >= sv.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: q0,
            n_qubits: sv.n_qubits,
        });
    }
    if q1 >= sv.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: q1,
            n_qubits: sv.n_qubits,
        });
    }
    if q0 == q1 {
        return Err(QuantumError::InvalidParameter {
            name: "q0 and q1 must differ".into(),
        });
    }

    let m0 = 1usize << q0;
    let m1 = 1usize << q1;
    let dim = sv.amps.len();

    let mut i = 0usize;
    while i < dim {
        // Only process groups once: when both q0=0 and q1=0
        if (i & m0) == 0 && (i & m1) == 0 {
            let i00 = i;
            let i01 = i | m1;
            let i10 = i | m0;
            let i11 = i | m0 | m1;

            let x0 = sv.amps[i00];
            let x1 = sv.amps[i01];
            let x2 = sv.amps[i10];
            let x3 = sv.amps[i11];

            sv.amps[i00] = gate[0][0] * x0 + gate[0][1] * x1 + gate[0][2] * x2 + gate[0][3] * x3;
            sv.amps[i01] = gate[1][0] * x0 + gate[1][1] * x1 + gate[1][2] * x2 + gate[1][3] * x3;
            sv.amps[i10] = gate[2][0] * x0 + gate[2][1] * x1 + gate[2][2] * x2 + gate[2][3] * x3;
            sv.amps[i11] = gate[3][0] * x0 + gate[3][1] * x1 + gate[3][2] * x2 + gate[3][3] * x3;
        }
        i += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statevec::state::StateVector;

    #[test]
    fn identity_2q_preserves_state() {
        let mut sv = StateVector::new_zero_state(2).expect(
            "n_qubits=2 is always a valid qubit count, so zero-state construction cannot fail",
        );
        let c1 = Complex32::new(1.0, 0.0);
        let c0 = Complex32::new(0.0, 0.0);
        let gate = [
            [c1, c0, c0, c0],
            [c0, c1, c0, c0],
            [c0, c0, c1, c0],
            [c0, c0, c0, c1],
        ];
        let orig = sv.amps.clone();
        apply_2q_inplace(&mut sv, 0, 1, &gate)
            .expect("q0=0 and q1=1 are distinct valid qubit indices within a 2-qubit state vector, so gate application cannot fail");
        for (a, b) in sv.amps.iter().zip(orig.iter()) {
            assert!((a - b).norm() < 1e-6);
        }
    }
}
