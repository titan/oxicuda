use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

/// Apply a single-qubit gate in-place using the bit-mask algorithm.
///
/// For each index pair (i0, i1) differing only in `qubit` bit, applies:
///   \[new_i0, new_i1\]^T = gate * \[amps\[i0\], amps\[i1\]\]^T
pub fn apply_1q_inplace(
    sv: &mut StateVector,
    qubit: usize,
    gate: &[[Complex32; 2]; 2],
) -> QuantumResult<()> {
    if qubit >= sv.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: qubit,
            n_qubits: sv.n_qubits,
        });
    }
    let mask = 1usize << qubit;
    let dim = sv.amps.len();

    let mut i = 0usize;
    while i < dim {
        if i & mask == 0 {
            let i1 = i | mask;
            let x0 = sv.amps[i];
            let x1 = sv.amps[i1];
            sv.amps[i] = gate[0][0] * x0 + gate[0][1] * x1;
            sv.amps[i1] = gate[1][0] * x0 + gate[1][1] * x1;
        }
        i += 1;
    }
    Ok(())
}

/// Apply a controlled single-qubit gate: gate is applied to `tgt` only when `ctrl` bit = 1.
pub fn apply_1q_controlled(
    sv: &mut StateVector,
    ctrl: usize,
    tgt: usize,
    gate: &[[Complex32; 2]; 2],
) -> QuantumResult<()> {
    if ctrl >= sv.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: ctrl,
            n_qubits: sv.n_qubits,
        });
    }
    if tgt >= sv.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: tgt,
            n_qubits: sv.n_qubits,
        });
    }
    if ctrl == tgt {
        return Err(QuantumError::InvalidParameter {
            name: "ctrl and tgt must differ".into(),
        });
    }

    let ctrl_mask = 1usize << ctrl;
    let tgt_mask = 1usize << tgt;
    let dim = sv.amps.len();

    let mut i = 0usize;
    while i < dim {
        // Only process when ctrl=1 and tgt=0 (to avoid double-processing)
        if (i & ctrl_mask) != 0 && (i & tgt_mask) == 0 {
            let i1 = i | tgt_mask;
            let x0 = sv.amps[i];
            let x1 = sv.amps[i1];
            sv.amps[i] = gate[0][0] * x0 + gate[0][1] * x1;
            sv.amps[i1] = gate[1][0] * x0 + gate[1][1] * x1;
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
    fn apply_x_gate_flips_zero_state() {
        let mut sv = StateVector::new_zero_state(1).expect(
            "1 is a valid qubit count; new_zero_state never fails for positive qubit counts",
        );
        let x_gate = [
            [Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)],
            [Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
        ];
        apply_1q_inplace(&mut sv, 0, &x_gate)
            .expect("qubit 0 is within range of a 1-qubit state vector");
        assert!((sv.amps[0].re).abs() < 1e-6);
        assert!((sv.amps[1].re - 1.0).abs() < 1e-6);
    }

    #[test]
    fn qubit_out_of_range_error() {
        let mut sv = StateVector::new_zero_state(2).expect("new_zero_state should succeed");
        let i_gate = [
            [Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
            [Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)],
        ];
        assert!(apply_1q_inplace(&mut sv, 5, &i_gate).is_err());
    }
}
