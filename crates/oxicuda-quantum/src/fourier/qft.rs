//! Quantum Fourier Transform (QFT) and its inverse.
//!
//! The QFT is the quantum analogue of the discrete Fourier transform (DFT).
//! Acting on a computational-basis state |j⟩ over `m` qubits it produces
//!
//! ```text
//! QFT |j⟩ = (1/√N) · Σ_{k=0}^{N-1} ω^{jk} |k⟩,   ω = e^{2πi/N},  N = 2^m.
//! ```
//!
//! Equivalently, for an arbitrary input amplitude vector `x`, the output
//! amplitude at basis index `k` is `(1/√N) · Σ_j x_j · ω^{jk}` — exactly the
//! (unnormalized-by-N, unitary) DFT matrix `F_{kj} = ω^{kj} / √N`.
//!
//! The circuit is the textbook Hadamard + controlled-phase ladder followed by a
//! bit-reversal swap network. With this crate's **little-endian** state vector
//! (`qubit q ↦ mask 1 << q`, so basis index `i = Σ_q b_q · 2^q`), the caller
//! passes the qubit indices ordered from least- to most-significant:
//! `qubits[0]` is the LSB and `qubits[m-1]` is the MSB.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::hadamard::gate_h;
use crate::gates::parametric::gate_phase;
use crate::statevec::apply_1q::{apply_1q_controlled, apply_1q_inplace};
use crate::statevec::state::StateVector;

/// Apply a controlled-phase rotation `diag(1, e^{iθ})` on `tgt` conditioned on `ctrl`.
///
/// The phase gate is diagonal, hence the controlled-phase interaction is
/// symmetric in `ctrl`/`tgt`; either ordering yields the same unitary. This is
/// the `R_k = diag(1, e^{iπ/2^{k}})` building block of the QFT ladder.
pub(crate) fn controlled_phase(
    sv: &mut StateVector,
    ctrl: usize,
    tgt: usize,
    angle: f32,
) -> QuantumResult<()> {
    apply_1q_controlled(sv, ctrl, tgt, &gate_phase(angle))
}

/// Validate that `qubits` is non-empty, in range, and free of duplicates.
fn validate_qubits(qubits: &[usize], n_qubits: usize) -> QuantumResult<()> {
    if qubits.is_empty() {
        return Err(QuantumError::EmptyInput);
    }
    for (pos, &q) in qubits.iter().enumerate() {
        if q >= n_qubits {
            return Err(QuantumError::QubitIndexOutOfRange { index: q, n_qubits });
        }
        // Reject duplicates: any later occurrence of the same index is invalid.
        if qubits[pos + 1..].contains(&q) {
            return Err(QuantumError::InvalidParameter {
                name: format!("duplicate qubit index {q}"),
            });
        }
    }
    Ok(())
}

/// Apply the Quantum Fourier Transform in place over `qubits`.
///
/// Convention: `qubits[0]` is the least-significant qubit and `qubits[m-1]` the
/// most-significant. After the Hadamard + controlled-phase ladder the routine
/// performs a bit-reversal swap network so that the output respects the same
/// little-endian basis ordering as the input.
///
/// The resulting transform satisfies, for any input amplitude vector,
/// `out[k] = (1/√N) · Σ_j in[j] · e^{2πi·jk/N}` with `N = 2^m`, where `j` and
/// `k` index the sub-register spanned by `qubits`.
///
/// # Errors
/// Returns [`QuantumError::EmptyInput`] if `qubits` is empty,
/// [`QuantumError::QubitIndexOutOfRange`] if any index is `>= sv.n_qubits`, and
/// [`QuantumError::InvalidParameter`] if `qubits` contains duplicates.
pub fn qft_inplace(sv: &mut StateVector, qubits: &[usize]) -> QuantumResult<()> {
    validate_qubits(qubits, sv.n_qubits)?;
    let m = qubits.len();

    // Hadamard + controlled-phase ladder, from the MSB down to the LSB.
    for i in (0..m).rev() {
        apply_1q_inplace(sv, qubits[i], &gate_h())?;
        for j in (0..i).rev() {
            let d = i - j;
            let angle = std::f32::consts::PI / 2f32.powi(d as i32);
            controlled_phase(sv, qubits[j], qubits[i], angle)?;
        }
    }

    // Bit-reversal: swap qubit k with qubit m-1-k.
    for k in 0..m / 2 {
        crate::gates::controlled::apply_swap(sv, qubits[k], qubits[m - 1 - k])?;
    }
    Ok(())
}

/// Apply the inverse Quantum Fourier Transform in place over `qubits`.
///
/// This is the exact adjoint of [`qft_inplace`]: the bit-reversal swap network
/// runs first, followed by the inverse Hadamard + controlled-phase ladder with
/// negated rotation angles. Applying [`qft_inplace`] and then
/// [`qft_inverse_inplace`] (with the same `qubits`) restores the original state
/// up to floating-point round-off.
///
/// # Errors
/// Same conditions as [`qft_inplace`].
pub fn qft_inverse_inplace(sv: &mut StateVector, qubits: &[usize]) -> QuantumResult<()> {
    validate_qubits(qubits, sv.n_qubits)?;
    let m = qubits.len();

    // Undo the bit-reversal first (swap is its own inverse).
    for k in 0..m / 2 {
        crate::gates::controlled::apply_swap(sv, qubits[k], qubits[m - 1 - k])?;
    }

    // Inverse ladder, from the LSB up to the MSB, with negated phases.
    for i in 0..m {
        for j in 0..i {
            let d = i - j;
            let angle = -std::f32::consts::PI / 2f32.powi(d as i32);
            controlled_phase(sv, qubits[j], qubits[i], angle)?;
        }
        apply_1q_inplace(sv, qubits[i], &gate_h())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use num_complex::Complex;
    use std::f32::consts::PI;

    type Complex32 = Complex<f32>;

    /// Brute-force unitary DFT reference: `out[k] = (1/√N) Σ_j x[j] · ω^{jk}`,
    /// with `ω = e^{2πi/N}`. Used to validate the circuit-level QFT.
    fn dft_reference(input: &[Complex32]) -> Vec<Complex32> {
        let n = input.len();
        let inv_sqrt_n = 1.0 / (n as f32).sqrt();
        let mut out = vec![Complex32::new(0.0, 0.0); n];
        for (k, slot) in out.iter_mut().enumerate() {
            let mut acc = Complex32::new(0.0, 0.0);
            for (j, &xj) in input.iter().enumerate() {
                let angle = 2.0 * PI * (j as f32) * (k as f32) / (n as f32);
                acc += xj * Complex32::new(angle.cos(), angle.sin());
            }
            *slot = acc * inv_sqrt_n;
        }
        out
    }

    #[test]
    fn qft_on_zero_state_is_uniform() {
        // QFT|000⟩ = (1/√8) Σ_k |k⟩ — every amplitude equals 1/√8 with zero phase.
        let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        qft_inplace(&mut sv, &[0, 1, 2]).expect("QFT on 3 qubits should succeed");
        let expected = 1.0 / (8.0_f32).sqrt();
        for (i, a) in sv.amps.iter().enumerate() {
            assert!((a.re - expected).abs() < 1e-5, "amp[{i}].re={}", a.re);
            assert!(a.im.abs() < 1e-5, "amp[{i}].im={}", a.im);
        }
        assert!(
            (sv.norm_sq() - 1.0).abs() < 1e-5,
            "norm_sq={}",
            sv.norm_sq()
        );
    }

    #[test]
    fn qft_then_inverse_is_identity_on_random_state() {
        // Fill a random normalized 3-qubit state, then check QFT⁻¹∘QFT = I.
        let mut rng = LcgRng::new(0xC0FFEE);
        let dim = 8;
        let mut amps = Vec::with_capacity(dim);
        for _ in 0..dim {
            amps.push(Complex32::new(rng.next_normal(), rng.next_normal()));
        }
        let mut sv = StateVector { amps, n_qubits: 3 };
        sv.normalize_inplace();
        let original = sv.amps.clone();

        qft_inplace(&mut sv, &[0, 1, 2]).expect("QFT on random 3-qubit state should succeed");
        qft_inverse_inplace(&mut sv, &[0, 1, 2])
            .expect("inverse QFT on 3-qubit state should succeed");

        for (i, (a, b)) in sv.amps.iter().zip(original.iter()).enumerate() {
            assert!(
                (a.re - b.re).abs() < 1e-5,
                "re mismatch at {i}: {a:?} vs {b:?}"
            );
            assert!(
                (a.im - b.im).abs() < 1e-5,
                "im mismatch at {i}: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn qft_matches_dft_matrix_n2() {
        // Input |01⟩ = index 1 (qubit 0 set). Expected via explicit DFT matrix.
        let mut sv = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &crate::gates::pauli::gate_x())
            .expect("X gate on qubit 0 should succeed");
        let input = sv.amps.clone();
        let reference = dft_reference(&input);

        qft_inplace(&mut sv, &[0, 1]).expect("QFT on 2 qubits should succeed");

        // Spot-checks against hand-computed values: ω = e^{iπ/2} = i, N = 4.
        let half = 0.5_f32;
        let checks = [
            (0usize, Complex32::new(half, 0.0)),
            (1, Complex32::new(0.0, half)),
            (2, Complex32::new(-half, 0.0)),
            (3, Complex32::new(0.0, -half)),
        ];
        for (k, want) in checks {
            assert!((sv.amps[k].re - want.re).abs() < 1e-5, "k={k} re");
            assert!((sv.amps[k].im - want.im).abs() < 1e-5, "k={k} im");
        }
        // Full agreement with the brute-force DFT reference.
        for (k, (a, r)) in sv.amps.iter().zip(reference.iter()).enumerate() {
            assert!((a.re - r.re).abs() < 1e-5, "k={k} re vs ref");
            assert!((a.im - r.im).abs() < 1e-5, "k={k} im vs ref");
        }
    }

    #[test]
    fn qft_matches_dft_matrix_n2_zero_input() {
        // Input |00⟩ ⇒ all outputs 0.5 + 0i.
        let mut sv = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        qft_inplace(&mut sv, &[0, 1]).expect("QFT on 2-qubit zero state should succeed");
        for (k, a) in sv.amps.iter().enumerate() {
            assert!((a.re - 0.5).abs() < 1e-5, "k={k} re={}", a.re);
            assert!(a.im.abs() < 1e-5, "k={k} im={}", a.im);
        }
    }

    #[test]
    fn qft_n3_single_excitation_spot_check() {
        // Input |001⟩ = index 1 ⇒ out[k] = (1/√8) ω^k, ω = e^{iπ/4}.
        let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &crate::gates::pauli::gate_x())
            .expect("X gate on qubit 0 should succeed");
        qft_inplace(&mut sv, &[0, 1, 2])
            .expect("QFT on 3-qubit single-excitation state should succeed");

        let s = 1.0 / (8.0_f32).sqrt();
        // k0 = s·ω^0 = s; k1 = s·ω^1 = s·(cos45+ i sin45) = 0.25 + 0.25i; k2 = s·ω^2 = s·i.
        assert!((sv.amps[0].re - s).abs() < 1e-5, "k0 re={}", sv.amps[0].re);
        assert!(sv.amps[0].im.abs() < 1e-5, "k0 im={}", sv.amps[0].im);
        assert!(
            (sv.amps[1].re - 0.25).abs() < 1e-5,
            "k1 re={}",
            sv.amps[1].re
        );
        assert!(
            (sv.amps[1].im - 0.25).abs() < 1e-5,
            "k1 im={}",
            sv.amps[1].im
        );
        assert!(sv.amps[2].re.abs() < 1e-5, "k2 re={}", sv.amps[2].re);
        assert!((sv.amps[2].im - s).abs() < 1e-5, "k2 im={}", sv.amps[2].im);
        assert!(
            (sv.norm_sq() - 1.0).abs() < 1e-5,
            "norm_sq={}",
            sv.norm_sq()
        );
    }

    #[test]
    fn qft_validation_rejects_bad_qubits() {
        let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        assert!(qft_inplace(&mut sv, &[]).is_err(), "empty must error");
        assert!(
            qft_inplace(&mut sv, &[0, 3]).is_err(),
            "out-of-range must error"
        );
        assert!(
            qft_inplace(&mut sv, &[0, 1, 1]).is_err(),
            "duplicate must error"
        );
        // Inverse path validates identically.
        assert!(qft_inverse_inplace(&mut sv, &[]).is_err());
        assert!(qft_inverse_inplace(&mut sv, &[5]).is_err());
        assert!(qft_inverse_inplace(&mut sv, &[2, 2]).is_err());
    }

    #[test]
    fn inverse_qft_of_uniform_state_is_zero_state() {
        // |+…+⟩ = (1/√N) Σ_k |k⟩ is QFT|0⟩, so QFT⁻¹ of it must return |0…0⟩.
        let n = 3;
        let dim = 1usize << n;
        let amp = 1.0 / (dim as f32).sqrt();
        let amps = vec![Complex32::new(amp, 0.0); dim];
        let mut sv = StateVector { amps, n_qubits: n };
        qft_inverse_inplace(&mut sv, &[0, 1, 2])
            .expect("inverse QFT on uniform state should succeed");
        assert!(
            (sv.amps[0].re - 1.0).abs() < 1e-5,
            "amp0 re={}",
            sv.amps[0].re
        );
        assert!(sv.amps[0].im.abs() < 1e-5, "amp0 im={}", sv.amps[0].im);
        for (k, a) in sv.amps.iter().enumerate().skip(1) {
            assert!(a.norm() < 1e-5, "amp[{k}]={a:?} should vanish");
        }
    }
}
