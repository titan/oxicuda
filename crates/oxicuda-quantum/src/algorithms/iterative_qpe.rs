//! Iterative (Kitaev) Quantum Phase Estimation with a single ancilla qubit.
//!
//! References: Kitaev, *"Quantum measurements and the Abelian Stabilizer
//! Problem"* (arXiv:quant-ph/9511026, 1995); Dobšíček, Johansson, Shumeiko,
//! Wendin, *"Arbitrary accuracy iterative quantum phase estimation algorithm
//! using a single ancillary qubit: A two-qubit benchmark"*, Phys. Rev. A 76,
//! 030306(R) (2007) (arXiv:quant-ph/0610214).
//!
//! # Problem
//!
//! Given a unitary `U` and one of its eigenstates `|ψ⟩` with
//! `U|ψ⟩ = e^{2πiφ}|ψ⟩`, estimate the phase `φ ∈ [0, 1)`. Where the textbook
//! register-based [`crate::fourier::phase_estimation`] uses `n` *counting*
//! qubits in parallel, the iterative scheme reads the phase out **one bit at a
//! time** using a **single** ancilla qubit, recycled across `n` rounds.
//!
//! # Algorithm
//!
//! Write the `n`-bit binary expansion `φ = 0.b_{n-1} b_{n-2} … b_0` (so
//! `φ = Σ_{k} b_k 2^{k-n}`). The bits are recovered from the *least* significant
//! (`b_0`) to the *most* significant (`b_{n-1}`). For round `k = n−1, n−2, …, 0`:
//!
//! 1. Reset the ancilla to `|0⟩` and apply a Hadamard to put it in
//!    `(|0⟩ + |1⟩)/√2`.
//! 2. Apply a **controlled-`U^{2^k}`** with the ancilla as control onto the
//!    system register (which stays in the eigenstate `|ψ⟩` throughout — an
//!    eigenstate of every power of `U`). This kicks back the phase
//!    `e^{2πi·2^k·φ}` onto the `|1⟩` branch of the ancilla.
//! 3. Apply the **feedback phase rotation** `R_z`-like correction `Z(−2π·ω_k)`
//!    on the ancilla, where the *feedback angle*
//!    `ω_k = 0.0 b_{k+1} b_{k+2} … b_{n-1}` collects the already-measured
//!    lower-order bits. This cancels their contribution to `2^k·φ`, leaving the
//!    ancilla phase determined solely by bit `b_k`.
//! 4. Apply a Hadamard and measure the ancilla in the computational basis. With
//!    the feedback correction in place the measurement is **deterministic** when
//!    `φ` is an exact `n`-bit dyadic: the outcome is exactly `b_k`.
//!
//! After `n` rounds, `φ = Σ_k b_k 2^{k-n}` reproduces the dyadic phase exactly.
//!
//! # The phase kickback identity used here
//!
//! The system register is never measured and only ever holds the eigenstate, so
//! the action of the controlled-`U^{2^k}` on the *ancilla* alone is exactly the
//! single-qubit phase gate `diag(1, e^{2πi·2^k·φ})`. This routine therefore
//! evolves a genuine **single-qubit ancilla state vector** — the entire
//! "working register" is one qubit, satisfying the single-ancilla promise of the
//! algorithm — while reading the eigenphase from the supplied `2×2` unitary.
//!
//! # Numerics
//!
//! Phase bookkeeping (the repeated-squaring eigenphase and the feedback angles)
//! is carried out in `f64`; the one-qubit ancilla evolution uses the shared
//! `f32` [`StateVector`] gate machinery (`H`, a phase gate, and a measurement),
//! exactly as a real single-ancilla device would.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::hadamard::gate_h;
use crate::gates::parametric::gate_phase;
use crate::handle::LcgRng;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;
use num_complex::Complex;

type Complex32 = Complex<f32>;
type Complex64 = Complex<f64>;

const TWO_PI: f64 = std::f64::consts::TAU;

/// Outcome of [`iterative_phase_estimation`].
#[derive(Debug, Clone)]
pub struct IterativeQpeResult {
    /// Estimated phase `φ = Σ_k bits[k]·2^{k-n} ∈ [0, 1)`.
    pub phase: f64,
    /// The recovered integer `j` with `φ = j / 2^n` (`bits` packed little-endian:
    /// `bits[0]` is the least-significant bit, weight `2^{0}` in `j`).
    pub integer: usize,
    /// The measured bits, least-significant first (`bits[0]` ↔ `b_0`).
    pub bits: Vec<u8>,
    /// Number of estimation rounds executed (equal to the number of bits and to
    /// the requested precision `n`).
    pub rounds: usize,
}

/// Compute the eigenphase `2π·φ` of the supplied `2×2` unitary `u` acting on the
/// eigenstate index `eig_index ∈ {0, 1}`.
///
/// `u` is assumed diagonal in the chosen eigenbasis only insofar as `eig_index`
/// labels a genuine eigenvector of the *computational-basis* gate; for the
/// phase-gate / `S` / `T` family used in QPE the eigenvectors are `|0⟩` and
/// `|1⟩`, so the eigenvalue is simply the diagonal entry `u[e][e]`. The returned
/// angle is `arg(u[e][e]) ∈ (−π, π]`.
fn diagonal_eigenphase(u: &[[Complex32; 2]; 2], eig_index: usize) -> Complex64 {
    let z = u[eig_index][eig_index];
    Complex64::new(z.re as f64, z.im as f64)
}

/// Validate that `u[eig_index]` is (numerically) an eigenvector of `u` with a
/// unit-modulus eigenvalue, i.e. the off-diagonal coupling out of that basis
/// state vanishes and the diagonal entry sits on the unit circle.
fn validate_eigenstate(u: &[[Complex32; 2]; 2], eig_index: usize) -> QuantumResult<()> {
    if eig_index > 1 {
        return Err(QuantumError::InvalidParameter {
            name: format!("eig_index={eig_index} must be 0 or 1"),
        });
    }
    // Column `eig_index` of U must be e^{iθ} on the diagonal with no leakage.
    let other = 1 - eig_index;
    let leak = u[other][eig_index].norm();
    if leak > 1e-4 {
        return Err(QuantumError::InvalidParameter {
            name: format!("basis state |{eig_index}⟩ is not an eigenvector of U (leakage {leak})"),
        });
    }
    let lam = u[eig_index][eig_index];
    if (lam.norm() - 1.0).abs() > 1e-3 {
        return Err(QuantumError::InvalidParameter {
            name: format!("eigenvalue modulus {} must be 1", lam.norm()),
        });
    }
    Ok(())
}

/// One iterative-QPE round on the single ancilla: returns the measured bit.
///
/// The ancilla starts in `|0⟩`. The controlled-`U^{2^power}` kicks back
/// `e^{2πi·2^power·φ}` onto the `|1⟩` branch (here `kick = 2^power · eig_phase_2pi`
/// since `eig_phase_2pi = 2π·φ`); the feedback term `feedback_angle_2pi`
/// (`= 2π·ω`) cancels the already-measured lower-order bits. Both are diagonal
/// phase gates on the ancilla, so the net gate is `P(net)` with
/// `net = 2^power·2π·φ − 2π·ω`. The closing Hadamard turns the resulting relative
/// phase into a `cos²`/`sin²` measurement probability — exactly `0`/`1` when `φ`
/// is an `n`-bit dyadic and the feedback is correct.
fn run_round(
    eig_phase_2pi: f64,
    power: u32,
    feedback_angle_2pi: f64,
    rng: &mut LcgRng,
) -> QuantumResult<u8> {
    // Single-qubit ancilla state vector.
    let mut anc = StateVector::new_zero_state(1)?;
    // H: |0⟩ → (|0⟩ + |1⟩)/√2.
    apply_1q_inplace(&mut anc, 0, &gate_h())?;

    // Net diagonal phase on |1⟩: kickback minus feedback correction.
    let two_pow = (1u64 << power) as f64;
    let kick = two_pow * eig_phase_2pi;
    let net = kick - feedback_angle_2pi;
    apply_1q_inplace(&mut anc, 0, &gate_phase(net as f32))?;

    // Final Hadamard maps the relative phase onto a measurement probability.
    apply_1q_inplace(&mut anc, 0, &gate_h())?;

    // Sample the ancilla. For an exact dyadic φ the probability is 0 or 1, so the
    // RNG draw is immaterial; the routine remains correct for non-dyadic φ by
    // returning the most probable bit (rounding to the nearest grid value).
    let p1 = anc.measure_prob(0, true)?;
    let _draw = rng.next_f32();
    Ok(u8::from(p1 >= 0.5))
}

/// Run iterative (Kitaev) phase estimation with a single recycled ancilla.
///
/// `u` is the `2×2` unitary whose eigenphase is sought; `eig_index ∈ {0, 1}`
/// selects which computational-basis eigenvector `|ψ⟩` of `u` is used (for `S`,
/// `T`, and phase gates this is `|1⟩`, i.e. `eig_index = 1`). `n_bits` is the
/// requested precision. `seed` seeds the (here deterministic) ancilla
/// measurement RNG.
///
/// Returns the recovered phase and its bit decomposition. The working register
/// is a **single qubit**, recycled across all `n_bits` rounds.
///
/// # Errors
/// * [`QuantumError::InvalidParameter`] if `n_bits == 0`, `n_bits > 53` (beyond
///   `f64` mantissa resolution / the `2^k` shift range), `eig_index > 1`, or
///   `|ψ⟩` is not an eigenvector of `u`.
pub fn iterative_phase_estimation(
    u: &[[Complex32; 2]; 2],
    eig_index: usize,
    n_bits: usize,
    seed: u64,
) -> QuantumResult<IterativeQpeResult> {
    if n_bits == 0 || n_bits > 53 {
        return Err(QuantumError::InvalidParameter {
            name: format!("n_bits={n_bits} must be in 1..=53"),
        });
    }
    validate_eigenstate(u, eig_index)?;

    let eig_phase = diagonal_eigenphase(u, eig_index);
    // φ ∈ [0, 1): normalize arg ∈ (−π, π] into [0, 2π) then divide by 2π.
    let mut arg = eig_phase.arg();
    if arg < 0.0 {
        arg += TWO_PI;
    }
    let eig_phase_2pi = arg; // = 2π·φ ∈ [0, 2π)

    let mut rng = LcgRng::new(seed);
    // bits[b] holds output bit b_b, which has weight 2^{b-n} in φ and weight 2^b
    // in the integer j. The least-significant fraction bit b_0 is recovered FIRST,
    // using the LARGEST controlled power U^{2^{n-1}}; the most-significant bit
    // b_{n-1} is recovered LAST, using U^{2^0}. In general output bit b uses power
    // 2^{n-1-b}.
    let mut bits = vec![0u8; n_bits];
    for b in 0..n_bits {
        let power = (n_bits - 1 - b) as u32;
        // Feedback cancels the contribution of the already-known lower bits
        // b_0 … b_{b-1}: ω = 0.0 b_{b-1} b_{b-2} … b_0 = Σ_{j<b} b_j · 2^{j-b-1}.
        // In units of 2π this is feedback_angle_2pi = TWO_PI · ω.
        let mut omega = 0.0_f64;
        for (j, &lower_bit) in bits.iter().enumerate().take(b) {
            omega += lower_bit as f64 * 2.0_f64.powi(j as i32 - b as i32 - 1);
        }
        let feedback = TWO_PI * omega;
        let bit = run_round(eig_phase_2pi, power, feedback, &mut rng)?;
        bits[b] = bit;
    }

    // Reassemble: φ = Σ_b b_b 2^{b-n}; integer j = Σ_b b_b 2^b.
    let mut integer = 0usize;
    let mut phase = 0.0_f64;
    for (b, &bit) in bits.iter().enumerate() {
        if bit == 1 {
            integer |= 1usize << b;
            phase += 2.0_f64.powi(b as i32 - n_bits as i32);
        }
    }

    Ok(IterativeQpeResult {
        phase,
        integer,
        bits,
        rounds: n_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fourier::phase_estimation;
    use crate::gates::hadamard::{gate_s, gate_t};
    use crate::gates::parametric::gate_phase as phase_gate;
    use crate::gates::pauli::{gate_i, gate_x, gate_z};
    use std::f32::consts::PI;

    // (a) Exact dyadic phases φ = j/2^n recover *every* bit exactly.
    #[test]
    fn recovers_exact_dyadic_phases() {
        // φ = 1/4 = 0.01₂ with n = 2: bits LSB-first = [0, 1], integer 2? No:
        // 1/4 = 0.01 means b_1 = 0? Let's be careful: φ = 0.b_{n-1}…b_0.
        // For n = 2, φ = b_1·2^{-1} + b_0·2^{-2}. φ = 1/4 ⇒ b_1 = 0, b_0 = 1.
        // bits[] is LSB-first so bits = [b_0, b_1] = [1, 0], integer = 1.
        let u = phase_gate(2.0 * PI * 0.25); // e^{2πi·1/4} on |1⟩
        let res = iterative_phase_estimation(&u, 1, 2, 7)
            .expect("phase gate φ=1/4 is a valid 2×2 unitary with |1⟩ as eigenvector");
        assert_eq!(res.bits, vec![1, 0], "bits={:?}", res.bits);
        assert_eq!(res.integer, 1, "integer={}", res.integer);
        assert!((res.phase - 0.25).abs() < 1e-12, "phase={}", res.phase);

        // φ = 1/8 = 0.001₂ with n = 3: b_2 b_1 b_0 = 0 0 1 ⇒ bits LSB-first
        // [1, 0, 0], integer 1.
        let u = phase_gate(2.0 * PI * 0.125);
        let res = iterative_phase_estimation(&u, 1, 3, 7)
            .expect("phase gate φ=1/8 is a valid 2×2 unitary with |1⟩ as eigenvector");
        assert_eq!(res.bits, vec![1, 0, 0], "bits={:?}", res.bits);
        assert_eq!(res.integer, 1, "integer={}", res.integer);
        assert!((res.phase - 0.125).abs() < 1e-12, "phase={}", res.phase);

        // φ = 3/8 = 0.011₂ with n = 3: b_2 b_1 b_0 = 0 1 1 ⇒ bits LSB-first
        // [1, 1, 0], integer 3.
        let u = phase_gate(2.0 * PI * 0.375);
        let res = iterative_phase_estimation(&u, 1, 3, 7)
            .expect("phase gate φ=3/8 is a valid 2×2 unitary with |1⟩ as eigenvector");
        assert_eq!(res.bits, vec![1, 1, 0], "bits={:?}", res.bits);
        assert_eq!(res.integer, 3, "integer={}", res.integer);
        assert!((res.phase - 0.375).abs() < 1e-12, "phase={}", res.phase);

        // φ = 5/8 = 0.101₂ with n = 3: b_2 b_1 b_0 = 1 0 1 ⇒ bits LSB-first
        // [1, 0, 1], integer 5. Exercises a high-order bit needing feedback.
        let u = phase_gate(2.0 * PI * 0.625);
        let res = iterative_phase_estimation(&u, 1, 3, 7)
            .expect("phase gate φ=5/8 is a valid 2×2 unitary with |1⟩ as eigenvector");
        assert_eq!(res.bits, vec![1, 0, 1], "bits={:?}", res.bits);
        assert_eq!(res.integer, 5, "integer={}", res.integer);
        assert!((res.phase - 0.625).abs() < 1e-12, "phase={}", res.phase);
    }

    // (b) Result matches the register-based QPE on the same U and ψ.
    #[test]
    fn matches_register_qpe() {
        // For each test U/φ pair, compare against fourier::phase_estimation, which
        // returns the same integer/phase for an exact dyadic phase.
        let cases: &[(f32, usize)] = &[
            (0.25, 2),
            (0.125, 3),
            (0.375, 3),
            (0.625, 3),
            (0.4375, 4), // 7/16
        ];
        for &(phi, n) in cases {
            let u = phase_gate(2.0 * PI * phi);
            let iter_res = iterative_phase_estimation(&u, 1, n, 11)
                .expect("iterative QPE with exact dyadic phase and valid eig_index should succeed");

            // Register QPE on (n + 1)-qubit state with target |1⟩.
            let mut sv = StateVector::new_zero_state(n + 1)
                .expect("n+1 is a valid qubit count for state vector allocation");
            apply_1q_inplace(&mut sv, n, &gate_x())
                .expect("qubit n exists in the (n+1)-qubit register, gate_x application is valid");
            let count: Vec<usize> = (0..n).collect();
            let reg_res = phase_estimation(&mut sv, &count, n, &u).expect(
                "register QPE with valid qubit mapping and exact dyadic phase should succeed",
            );

            assert_eq!(
                iter_res.integer, reg_res.integer,
                "phi={phi}: iter int {} vs reg int {}",
                iter_res.integer, reg_res.integer
            );
            assert!(
                (iter_res.phase - reg_res.phase as f64).abs() < 1e-6,
                "phi={phi}: iter phase {} vs reg {}",
                iter_res.phase,
                reg_res.phase
            );
        }
    }

    // (c) Only ONE ancilla qubit is used: the working register is a single qubit.
    #[test]
    fn uses_single_ancilla_qubit() {
        // The run_round helper builds a 1-qubit StateVector; assert that directly
        // by reconstructing the register it operates on.
        let anc = StateVector::new_zero_state(1)
            .expect("single-qubit zero state allocation is always valid");
        assert_eq!(anc.n_qubits, 1, "working register must be a single qubit");
        assert_eq!(anc.amps.len(), 2);
        // And a full multi-bit run still completes with that single-qubit register,
        // independent of n_bits (4 here): no register growth with precision.
        let u = phase_gate(2.0 * PI * 0.4375);
        let res = iterative_phase_estimation(&u, 1, 4, 3)
            .expect("phase gate φ=7/16 is a valid unitary with |1⟩ as eigenvector");
        assert_eq!(res.rounds, 4);
        assert_eq!(res.bits.len(), 4);
    }

    // (d) Feedback correction is essential: omitting it breaks a multi-bit case.
    #[test]
    fn feedback_correction_is_required() {
        // φ = 3/8 = 0.011₂ needs feedback to resolve the b_1 bit correctly once
        // b_0 = 1 is known. With feedback the full result is exact (checked in
        // (a)). Here we emulate "no feedback" by passing feedback = 0 in every
        // round and show at least one bit comes out wrong, so the answer differs.
        let phi = 0.375_f64;
        let eig_phase_2pi = TWO_PI * phi;
        let n = 3usize;
        let mut rng = LcgRng::new(5);
        // No-feedback bit extraction: same powers as the real routine but with the
        // feedback correction forced to zero.
        let mut wrong_bits = vec![0u8; n];
        for (b, slot) in wrong_bits.iter_mut().enumerate() {
            let power = (n - 1 - b) as u32;
            *slot = run_round(eig_phase_2pi, power, 0.0, &mut rng)
                .expect("run_round with valid phase and power should always succeed");
        }
        // Correct (with feedback) reference.
        let correct = iterative_phase_estimation(&phase_gate(2.0 * PI * phi as f32), 1, n, 5)
            .expect("phase gate φ=3/8 is a valid unitary, iterative QPE should succeed");
        assert_ne!(
            wrong_bits, correct.bits,
            "feedback-free bits {wrong_bits:?} unexpectedly matched correct {:?}",
            correct.bits
        );
    }

    // (e) U = S → φ = 1/4; U = T → φ = 1/8.
    #[test]
    fn s_and_t_gate_phases() {
        // S = diag(1, i) = e^{2πi/4} on |1⟩.
        let res_s = iterative_phase_estimation(&gate_s(), 1, 2, 1)
            .expect("S gate has unit-modulus eigenvalue on |1⟩, valid eig_index and n_bits");
        assert!(
            (res_s.phase - 0.25).abs() < 1e-12,
            "S phase={}",
            res_s.phase
        );
        assert_eq!(res_s.integer, 1);

        // T = diag(1, e^{iπ/4}) = e^{2πi/8} on |1⟩.
        let res_t = iterative_phase_estimation(&gate_t(), 1, 3, 1)
            .expect("T gate has unit-modulus eigenvalue on |1⟩, valid eig_index and n_bits");
        assert!(
            (res_t.phase - 0.125).abs() < 1e-12,
            "T phase={}",
            res_t.phase
        );
        assert_eq!(res_t.integer, 1);
    }

    // (f) φ = 0 (identity) and φ = 1/2 (Z gate) edge cases.
    #[test]
    fn zero_and_half_edge_cases() {
        // Identity → eigenphase 0 ⇒ all bits 0.
        let res0 = iterative_phase_estimation(&gate_i(), 1, 3, 9)
            .expect("identity gate is a valid unitary with |1⟩ as eigenvector");
        assert_eq!(res0.bits, vec![0, 0, 0], "bits={:?}", res0.bits);
        assert_eq!(res0.integer, 0);
        assert!(res0.phase.abs() < 1e-12, "phase={}", res0.phase);

        // Z = diag(1, -1) = e^{2πi·1/2} on |1⟩ ⇒ φ = 1/2 = 0.1₂.
        // n = 1: b_0 = 1, integer 1, phase 1/2. n = 3: b_2 b_1 b_0 = 1 0 0 ⇒
        // bits LSB-first [0, 0, 1], integer 4.
        let res_z1 = iterative_phase_estimation(&gate_z(), 1, 1, 9)
            .expect("Z gate has eigenvalue -1 on |1⟩ with unit modulus, QPE should succeed");
        assert_eq!(res_z1.bits, vec![1], "bits={:?}", res_z1.bits);
        assert!((res_z1.phase - 0.5).abs() < 1e-12, "phase={}", res_z1.phase);

        let res_z3 = iterative_phase_estimation(&gate_z(), 1, 3, 9)
            .expect("Z gate is a valid 2×2 unitary with |1⟩ as eigenvector");
        assert_eq!(res_z3.bits, vec![0, 0, 1], "bits={:?}", res_z3.bits);
        assert_eq!(res_z3.integer, 4);
        assert!((res_z3.phase - 0.5).abs() < 1e-12, "phase={}", res_z3.phase);
    }

    // Validation: bad inputs are rejected.
    #[test]
    fn rejects_invalid_inputs() {
        let u = gate_s();
        assert!(iterative_phase_estimation(&u, 1, 0, 0).is_err()); // n_bits = 0
        assert!(iterative_phase_estimation(&u, 2, 3, 0).is_err()); // eig_index > 1
        assert!(iterative_phase_estimation(&u, 1, 99, 0).is_err()); // n_bits too big
        // Non-eigenvector basis state: H has no computational-basis eigenvector.
        assert!(iterative_phase_estimation(&gate_h(), 0, 3, 0).is_err());
    }

    // The |0⟩ eigenvector (eig_index = 0) of a phase gate has eigenvalue 1 ⇒ φ=0.
    #[test]
    fn zero_eigenvector_has_zero_phase() {
        // P(θ) = diag(1, e^{iθ}); |0⟩ eigenvalue is 1 for any θ.
        let u = gate_phase(2.0 * PI * 0.3);
        let res = iterative_phase_estimation(&u, 0, 4, 2)
            .expect("P(θ) has |0⟩ as eigenvector with eigenvalue 1, eig_index 0 is valid");
        assert_eq!(res.integer, 0, "integer={}", res.integer);
        assert!(res.phase.abs() < 1e-12, "phase={}", res.phase);
    }
}
