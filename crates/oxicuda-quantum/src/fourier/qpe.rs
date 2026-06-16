//! Quantum Phase Estimation (QPE).
//!
//! Given a unitary `U` and one of its eigenstates `|ψ⟩` with `U|ψ⟩ = e^{2πiφ}|ψ⟩`,
//! QPE estimates the phase `φ ∈ [0, 1)` to `n`-bit precision using a register of
//! `n` *counting* qubits and a single *target* qubit prepared in `|ψ⟩`.
//!
//! The circuit is the canonical one:
//! 1. Hadamard every counting qubit (uniform superposition).
//! 2. For counting qubit `k`, apply controlled-`U^{2^k}` onto the target. The
//!    powers are formed by repeated squaring, `U^{2^{k+1}} = (U^{2^k})²`.
//! 3. Apply the inverse QFT to the counting register.
//! 4. Read out the counting register; the most probable basis value `j` gives
//!    `φ ≈ j / 2^n`.
//!
//! The counting register is interpreted **little-endian** to match
//! [`crate::fourier::qft::qft_inverse_inplace`]: `count_qubits[0]` is the
//! least-significant bit of the estimated integer. The caller is responsible for
//! preparing `target_qubit` in an eigenstate of `u` before calling.

use crate::error::{QuantumError, QuantumResult};
use crate::fourier::qft::qft_inverse_inplace;
use crate::gates::hadamard::gate_h;
use crate::statevec::apply_1q::{apply_1q_controlled, apply_1q_inplace};
use crate::statevec::state::StateVector;
use num_complex::Complex;

type Complex32 = Complex<f32>;

/// Outcome of [`phase_estimation`].
#[derive(Debug, Clone)]
pub struct PhaseEstimationResult {
    /// Estimated phase `φ = integer / 2^n ∈ [0, 1)`.
    pub phase: f32,
    /// Most probable counting-register integer `j`.
    pub integer: usize,
    /// Probability mass on the winning integer (peak height of the readout).
    pub probability: f32,
    /// The argmax basis value of the counting register (identical to `integer`;
    /// retained as an explicit field for callers that distinguish the readout
    /// histogram's argmax from the derived phase integer).
    pub counts_argmax: usize,
}

/// Complex `2×2` matrix product `a · b` (row-major).
fn mat2_mul(a: &[[Complex32; 2]; 2], b: &[[Complex32; 2]; 2]) -> [[Complex32; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Validate the counting register and target qubit for [`phase_estimation`].
fn validate(count_qubits: &[usize], target_qubit: usize, n_qubits: usize) -> QuantumResult<()> {
    if count_qubits.is_empty() {
        return Err(QuantumError::EmptyInput);
    }
    if target_qubit >= n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: target_qubit,
            n_qubits,
        });
    }
    for (pos, &q) in count_qubits.iter().enumerate() {
        if q >= n_qubits {
            return Err(QuantumError::QubitIndexOutOfRange { index: q, n_qubits });
        }
        if q == target_qubit {
            return Err(QuantumError::InvalidParameter {
                name: format!("counting qubit {q} collides with target"),
            });
        }
        if count_qubits[pos + 1..].contains(&q) {
            return Err(QuantumError::InvalidParameter {
                name: format!("duplicate counting qubit index {q}"),
            });
        }
    }
    Ok(())
}

/// Run Quantum Phase Estimation, returning the most probable phase estimate.
///
/// The caller must pre-prepare `sv` so that `target_qubit` holds an eigenstate of
/// `u`; the counting qubits should be in `|0⟩` (this routine applies their
/// Hadamards). `count_qubits[0]` is treated as the least-significant bit of the
/// estimated integer.
///
/// # Errors
/// Returns [`QuantumError::EmptyInput`] if `count_qubits` is empty,
/// [`QuantumError::QubitIndexOutOfRange`] if any counting qubit or the target is
/// out of range, and [`QuantumError::InvalidParameter`] if a counting qubit
/// duplicates another or collides with the target.
pub fn phase_estimation(
    sv: &mut StateVector,
    count_qubits: &[usize],
    target_qubit: usize,
    u: &[[Complex32; 2]; 2],
) -> QuantumResult<PhaseEstimationResult> {
    validate(count_qubits, target_qubit, sv.n_qubits)?;
    let n = count_qubits.len();

    // Step 1: superpose every counting qubit.
    for &q in count_qubits {
        apply_1q_inplace(sv, q, &gate_h())?;
    }

    // Step 2: controlled-U^{2^k} ladder via repeated squaring.
    let mut u_pow = *u;
    for &q in count_qubits.iter().take(n) {
        apply_1q_controlled(sv, q, target_qubit, &u_pow)?;
        u_pow = mat2_mul(&u_pow, &u_pow);
    }

    // Step 3: inverse QFT on the counting register (little-endian).
    qft_inverse_inplace(sv, count_qubits)?;

    // Step 4: marginal readout over the counting register.
    let dim = sv.amps.len();
    let n_outcomes = 1usize << n;
    let mut probs = vec![0.0_f32; n_outcomes];
    for (i, a) in sv.amps.iter().enumerate().take(dim) {
        let mut c = 0usize;
        for (k, &q) in count_qubits.iter().enumerate() {
            c |= ((i >> q) & 1) << k;
        }
        probs[c] += a.norm_sqr();
    }

    let mut integer = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (c, &p) in probs.iter().enumerate() {
        if p > best {
            best = p;
            integer = c;
        }
    }

    let phase = integer as f32 / 2f32.powi(n as i32);
    Ok(PhaseEstimationResult {
        phase,
        integer,
        probability: best,
        counts_argmax: integer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::hadamard::{gate_s, gate_t};
    use crate::gates::parametric::{gate_phase, gate_rz};
    use crate::gates::pauli::{gate_i, gate_x};
    use std::f32::consts::PI;

    /// Prepare an (n+1)-qubit register with the target qubit flipped to |1⟩.
    fn prepared_state(n_count: usize, target: usize) -> StateVector {
        let mut sv = StateVector::new_zero_state(n_count + 1)
            .expect("valid qubit count for QPE test register");
        apply_1q_inplace(&mut sv, target, &gate_x()).expect("X gate applied to valid target qubit");
        sv
    }

    #[test]
    fn qpe_phase_one_quarter() {
        // U = S = diag(1, i) = e^{2πi/4} on |1⟩ ⇒ φ = 1/4 with n = 2 ⇒ integer 1.
        let mut sv = prepared_state(2, 2);
        let res = phase_estimation(&mut sv, &[0, 1], 2, &gate_s()).expect("phase estimation with valid non-overlapping count_qubits [0,1] and target 2 on a 3-qubit state should succeed");
        assert_eq!(res.integer, 1, "integer={}", res.integer);
        assert!((res.phase - 0.25).abs() < 1e-6, "phase={}", res.phase);
        assert!(res.probability > 0.999, "prob={}", res.probability);
        assert_eq!(res.counts_argmax, res.integer);
    }

    #[test]
    fn qpe_phase_one_eighth() {
        // U = T = diag(1, e^{iπ/4}) = e^{2πi/8} on |1⟩ ⇒ φ = 1/8 with n = 3 ⇒ integer 1.
        let mut sv = prepared_state(3, 3);
        let res = phase_estimation(&mut sv, &[0, 1, 2], 3, &gate_t()).expect("phase estimation with valid non-overlapping count_qubits [0,1,2] and target 3 on a 4-qubit state should succeed");
        assert_eq!(res.integer, 1, "integer={}", res.integer);
        assert!((res.phase - 0.125).abs() < 1e-6, "phase={}", res.phase);
        assert!(res.probability > 0.999, "prob={}", res.probability);
    }

    #[test]
    fn qpe_phase_three_eighths() {
        // U = P(3π/4) = e^{2πi·3/8} on |1⟩ ⇒ φ = 3/8 with n = 3 ⇒ integer 3.
        let mut sv = prepared_state(3, 3);
        let u = gate_phase(3.0 * PI / 4.0);
        let res = phase_estimation(&mut sv, &[0, 1, 2], 3, &u).expect("phase estimation with valid non-overlapping count_qubits [0,1,2] and target 3 on a 4-qubit state should succeed");
        assert_eq!(res.integer, 3, "integer={}", res.integer);
        assert!((res.phase - 0.375).abs() < 1e-6, "phase={}", res.phase);
        assert!(res.probability > 0.999, "prob={}", res.probability);
    }

    #[test]
    fn qpe_identity_phase_zero() {
        // Identity has eigenphase 0 ⇒ integer 0, phase 0, probability ≈ 1.
        let mut sv = prepared_state(2, 2);
        let res =
            phase_estimation(&mut sv, &[0, 1], 2, &gate_i()).expect("value should be present");
        assert_eq!(res.integer, 0, "integer={}", res.integer);
        assert!(res.phase.abs() < 1e-6, "phase={}", res.phase);
        assert!(
            (res.probability - 1.0).abs() < 1e-4,
            "prob={}",
            res.probability
        );
    }

    #[test]
    fn qpe_validation_rejects_bad_inputs() {
        let mut sv = prepared_state(2, 2);
        // Empty counting register.
        assert!(phase_estimation(&mut sv, &[], 2, &gate_s()).is_err());
        // Target inside the counting register.
        assert!(phase_estimation(&mut sv, &[0, 1], 1, &gate_s()).is_err());
        // Out-of-range counting qubit.
        assert!(phase_estimation(&mut sv, &[0, 9], 2, &gate_s()).is_err());
        // Out-of-range target.
        assert!(phase_estimation(&mut sv, &[0, 1], 9, &gate_s()).is_err());
    }

    #[test]
    fn qpe_rz_eigenphase() {
        // Rz(π) = diag(e^{-iπ/2}, e^{+iπ/2}); on |1⟩ the eigenvalue is e^{+iπ/2} = i,
        // i.e. e^{2πi·(1/4)} ⇒ φ = 1/4 with n = 2 ⇒ integer 1.
        let mut sv = prepared_state(2, 2);
        let res =
            phase_estimation(&mut sv, &[0, 1], 2, &gate_rz(PI)).expect("value should be present");
        assert_eq!(res.integer, 1, "integer={}", res.integer);
        assert!((res.phase - 0.25).abs() < 1e-6, "phase={}", res.phase);
    }
}
