//! Mid-circuit measurement with classical feed-forward conditional gates.
//!
//! This module operates in the **state-vector** domain (distinct from the
//! stabilizer-tableau measurement in [`crate::stabilizer`]). It provides:
//!
//! * [`ClassicalRegister`] — a register of optionally-set classical bits that
//!   records measurement outcomes.
//! * [`measure_and_collapse`] — sample a qubit, collapse the [`StateVector`] to
//!   the measured branch, renormalize, and store the bit.
//! * [`measure_deterministic`] — project the state onto a *forced* outcome
//!   (useful for constructing deterministic test scenarios).
//! * [`apply_if`] — apply a 1-qubit gate only when a predicate over recorded
//!   classical bits is satisfied (classical feed-forward).
//! * [`run`] — a small executor over a list of [`MidCircuitOp`] producing the
//!   final state and classical register.

use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::handle::LcgRng;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

/// A register of classical bits, each either unmeasured (`None`) or set.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassicalRegister {
    bits: Vec<Option<bool>>,
}

impl ClassicalRegister {
    /// Create a register of `len` unmeasured bits.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            bits: vec![None; len],
        }
    }

    /// Number of classical bits.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Whether the register has zero bits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Read bit `idx` (returns `None` if unmeasured).
    ///
    /// # Errors
    /// [`QuantumError::InvalidParameter`] if `idx` is out of range.
    pub fn get(&self, idx: usize) -> QuantumResult<Option<bool>> {
        match self.bits.get(idx) {
            Some(b) => Ok(*b),
            None => Err(QuantumError::InvalidParameter {
                name: format!(
                    "classical bit index {idx} out of range (len {})",
                    self.bits.len()
                ),
            }),
        }
    }

    /// Write bit `idx`.
    ///
    /// # Errors
    /// [`QuantumError::InvalidParameter`] if `idx` is out of range.
    pub fn set(&mut self, idx: usize, value: bool) -> QuantumResult<()> {
        match self.bits.get_mut(idx) {
            Some(slot) => {
                *slot = Some(value);
                Ok(())
            }
            None => Err(QuantumError::InvalidParameter {
                name: format!(
                    "classical bit index {idx} out of range (len {})",
                    self.bits.len()
                ),
            }),
        }
    }
}

/// Project `state` onto the subspace where `qubit == outcome`, then renormalize.
///
/// Returns the post-measurement squared norm before renormalization (i.e. the
/// probability of that outcome) so callers can detect a near-zero branch.
fn project_outcome(state: &mut StateVector, qubit: usize, outcome: bool) -> QuantumResult<f32> {
    if qubit >= state.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: qubit,
            n_qubits: state.n_qubits,
        });
    }
    let mask = 1usize << qubit;
    let mut norm_sq = 0.0f32;
    for (i, a) in state.amps.iter_mut().enumerate() {
        let bit_set = (i & mask) != 0;
        if bit_set != outcome {
            *a = Complex32::new(0.0, 0.0);
        } else {
            norm_sq += a.norm_sqr();
        }
    }
    let norm = norm_sq.sqrt();
    if norm < 1e-12 {
        return Err(QuantumError::MeasurementFailed);
    }
    let inv = 1.0 / norm;
    for a in &mut state.amps {
        *a *= inv;
    }
    Ok(norm_sq)
}

/// Sample a measurement of `qubit`, collapse `state` in place to the measured
/// branch (renormalizing), and record the outcome in `creg[creg_idx]`.
///
/// # Errors
/// * [`QuantumError::QubitIndexOutOfRange`] if `qubit` is out of range.
/// * [`QuantumError::InvalidParameter`] if `creg_idx` is out of range.
/// * [`QuantumError::MeasurementFailed`] if the sampled branch has ~0 norm.
pub fn measure_and_collapse(
    state: &mut StateVector,
    qubit: usize,
    rng: &mut LcgRng,
    creg: &mut ClassicalRegister,
    creg_idx: usize,
) -> QuantumResult<bool> {
    if qubit >= state.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: qubit,
            n_qubits: state.n_qubits,
        });
    }
    if creg_idx >= creg.len() {
        return Err(QuantumError::InvalidParameter {
            name: format!("creg index {creg_idx} out of range (len {})", creg.len()),
        });
    }
    let p1 = state.measure_prob(qubit, true)?;
    let r = rng.next_f32();
    let outcome = r < p1;
    project_outcome(state, qubit, outcome)?;
    creg.set(creg_idx, outcome)?;
    Ok(outcome)
}

/// Project `state` onto a **forced** outcome of `qubit` and renormalize.
///
/// Intended for deterministic test construction.
///
/// # Errors
/// * [`QuantumError::QubitIndexOutOfRange`] if `qubit` is out of range.
/// * [`QuantumError::MeasurementFailed`] if `forced_outcome` has ~0 probability.
pub fn measure_deterministic(
    state: &mut StateVector,
    qubit: usize,
    forced_outcome: bool,
) -> QuantumResult<()> {
    project_outcome(state, qubit, forced_outcome)?;
    Ok(())
}

/// Apply `gate` to `target` iff every `(creg_idx, expected_bit)` in `predicate`
/// matches the recorded classical bit.
///
/// # Errors
/// * [`QuantumError::InvalidParameter`] if a referenced classical bit is
///   unmeasured (`None`) or out of range.
/// * [`QuantumError::QubitIndexOutOfRange`] if `target` is out of range.
pub fn apply_if(
    state: &mut StateVector,
    creg: &ClassicalRegister,
    predicate: &[(usize, bool)],
    gate: &[[Complex32; 2]; 2],
    target: usize,
) -> QuantumResult<()> {
    if target >= state.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: target,
            n_qubits: state.n_qubits,
        });
    }
    let mut all_match = true;
    for &(idx, expected) in predicate {
        match creg.get(idx)? {
            Some(bit) => {
                if bit != expected {
                    all_match = false;
                }
            }
            None => {
                return Err(QuantumError::InvalidParameter {
                    name: format!("conditional references unmeasured classical bit {idx}"),
                });
            }
        }
    }
    if all_match {
        apply_1q_inplace(state, target, gate)?;
    }
    Ok(())
}

/// One operation in a mid-circuit program executed by [`run`].
#[derive(Debug, Clone)]
pub enum MidCircuitOp {
    /// Apply a 1-qubit `gate` to `qubit`.
    Gate1q {
        /// The 2×2 gate matrix.
        gate: [[Complex32; 2]; 2],
        /// Target qubit.
        qubit: usize,
    },
    /// Measure `qubit` and store the result in `creg[creg_idx]`.
    Measure {
        /// Qubit to measure.
        qubit: usize,
        /// Classical register slot to record into.
        creg_idx: usize,
    },
    /// Apply `gate` to `target` iff `predicate` matches the classical register.
    Conditional {
        /// Predicate of `(creg_idx, expected_bit)` constraints (all must hold).
        predicate: Vec<(usize, bool)>,
        /// The 2×2 gate matrix.
        gate: [[Complex32; 2]; 2],
        /// Target qubit.
        target: usize,
    },
}

/// Execute a list of [`MidCircuitOp`] starting from |0…0⟩.
///
/// The classical register is sized to hold the largest `creg_idx` referenced by
/// any [`MidCircuitOp::Measure`] or [`MidCircuitOp::Conditional`] (at least 1).
///
/// # Errors
/// Propagates any error from the individual operations, and returns
/// [`QuantumError::InvalidQubitCount`] for an invalid `n_qubits`.
pub fn run(
    ops: &[MidCircuitOp],
    n_qubits: usize,
    rng: &mut LcgRng,
) -> QuantumResult<(StateVector, ClassicalRegister)> {
    let mut state = StateVector::new_zero_state(n_qubits)?;

    // Determine register size from referenced indices.
    let mut creg_len = 0usize;
    for op in ops {
        match op {
            MidCircuitOp::Measure { creg_idx, .. } => {
                creg_len = creg_len.max(creg_idx + 1);
            }
            MidCircuitOp::Conditional { predicate, .. } => {
                for &(idx, _) in predicate {
                    creg_len = creg_len.max(idx + 1);
                }
            }
            MidCircuitOp::Gate1q { .. } => {}
        }
    }
    let mut creg = ClassicalRegister::new(creg_len.max(1));

    for op in ops {
        match op {
            MidCircuitOp::Gate1q { gate, qubit } => {
                apply_1q_inplace(&mut state, *qubit, gate)?;
            }
            MidCircuitOp::Measure { qubit, creg_idx } => {
                measure_and_collapse(&mut state, *qubit, rng, &mut creg, *creg_idx)?;
            }
            MidCircuitOp::Conditional {
                predicate,
                gate,
                target,
            } => {
                apply_if(&mut state, &creg, predicate, gate, *target)?;
            }
        }
    }
    Ok((state, creg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::controlled::apply_cnot;
    use crate::gates::hadamard::gate_h;
    use crate::gates::pauli::gate_x;
    use crate::statevec::apply_1q::apply_1q_inplace;

    fn bell_pair() -> StateVector {
        let mut sv = StateVector::new_zero_state(2).expect("2 is a valid qubit count");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("qubit 0 is within 2-qubit state");
        apply_cnot(&mut sv, 0, 1).expect("qubits 0 and 1 are within 2-qubit state");
        sv
    }

    #[test]
    fn t01_measure_zero_gives_zero_unchanged() {
        let mut sv = StateVector::new_zero_state(1).expect("1 is a valid qubit count");
        let before = sv.amps.clone();
        let mut rng = LcgRng::new(1);
        let mut creg = ClassicalRegister::new(1);
        let outcome = measure_and_collapse(&mut sv, 0, &mut rng, &mut creg, 0)
            .expect("qubit 0 and creg index 0 are in range");
        assert!(!outcome, "|0> must measure to 0");
        for (a, b) in sv.amps.iter().zip(before.iter()) {
            assert!((a - b).norm() < 1e-6);
        }
        assert_eq!(
            creg.get(0).expect("index 0 is within 1-bit register"),
            Some(false)
        );
    }

    #[test]
    fn t02_measure_plus_collapses_norm_one() {
        let mut sv = StateVector::new_zero_state(1).expect("1 is a valid qubit count");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("qubit 0 is within 1-qubit state");
        let mut rng = LcgRng::new(5);
        let mut creg = ClassicalRegister::new(1);
        let outcome = measure_and_collapse(&mut sv, 0, &mut rng, &mut creg, 0)
            .expect("qubit 0 and creg index 0 are in range");
        // Post-measure norm == 1, and state is a basis state.
        assert!((sv.norm_sq() - 1.0).abs() < 1e-5);
        let idx = usize::from(outcome);
        assert!((sv.amps[idx].norm() - 1.0).abs() < 1e-5);
        assert!(sv.amps[1 - idx].norm() < 1e-5);
    }

    #[test]
    fn t03_bell_feedforward_correlation() {
        // Measuring one half of a Bell pair determines the other qubit:
        // verify via measure_prob on the partner.
        let mut sv = bell_pair();
        let mut rng = LcgRng::new(123);
        let mut creg = ClassicalRegister::new(1);
        let outcome = measure_and_collapse(&mut sv, 0, &mut rng, &mut creg, 0)
            .expect("qubit 0 and creg index 0 are in range");
        // The partner (qubit 1) must now be deterministic and equal to outcome.
        let p_same = sv
            .measure_prob(1, outcome)
            .expect("qubit 1 is within 2-qubit Bell state");
        let p_other = sv
            .measure_prob(1, !outcome)
            .expect("qubit 1 is within 2-qubit Bell state");
        assert!((p_same - 1.0).abs() < 1e-5, "p_same={p_same}");
        assert!(p_other < 1e-5, "p_other={p_other}");
    }

    #[test]
    fn t04_conditional_x_applied_when_bit_one() {
        // Teleportation-style correction toy: force qubit 0 = |1>, measure it,
        // then conditional-X on qubit 1 flips |0> -> |1>.
        let mut sv = StateVector::new_zero_state(2).expect("2 is a valid qubit count");
        apply_1q_inplace(&mut sv, 0, &gate_x()).expect("qubit 0 is within 2-qubit state");
        let mut rng = LcgRng::new(2);
        let mut creg = ClassicalRegister::new(1);
        let outcome = measure_and_collapse(&mut sv, 0, &mut rng, &mut creg, 0)
            .expect("qubit 0 and creg index 0 are in range");
        assert!(outcome);
        apply_if(&mut sv, &creg, &[(0, true)], &gate_x(), 1)
            .expect("bit 0 is measured and qubit 1 is in range");
        // q1 should now be |1>: amplitude at index 0b11 = 3.
        assert!((sv.amps[3].norm() - 1.0).abs() < 1e-5, "amps={:?}", sv.amps);
    }

    #[test]
    fn t05_creg_stores_outcomes() {
        let mut sv = StateVector::new_zero_state(2).expect("2 is a valid qubit count");
        apply_1q_inplace(&mut sv, 1, &gate_x()).expect("qubit 1 is within 2-qubit state");
        let mut rng = LcgRng::new(3);
        let mut creg = ClassicalRegister::new(2);
        let o0 = measure_and_collapse(&mut sv, 0, &mut rng, &mut creg, 0)
            .expect("qubit 0 and creg index 0 are in range");
        let o1 = measure_and_collapse(&mut sv, 1, &mut rng, &mut creg, 1)
            .expect("qubit 1 and creg index 1 are in range");
        assert_eq!(
            creg.get(0).expect("index 0 is within 2-bit register"),
            Some(o0)
        );
        assert_eq!(
            creg.get(1).expect("index 1 is within 2-bit register"),
            Some(o1)
        );
        assert!(!o0);
        assert!(o1);
    }

    #[test]
    fn t06_idempotent_second_measurement() {
        let mut sv = StateVector::new_zero_state(1).expect("1 is a valid qubit count");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("qubit 0 is within 1-qubit state");
        let mut rng = LcgRng::new(77);
        let mut creg = ClassicalRegister::new(2);
        let first = measure_and_collapse(&mut sv, 0, &mut rng, &mut creg, 0)
            .expect("qubit 0 and creg index 0 are in range");
        // Measuring an already-collapsed qubit yields the same bit with prob 1.
        let p_first = sv
            .measure_prob(0, first)
            .expect("qubit 0 is within 1-qubit collapsed state");
        assert!((p_first - 1.0).abs() < 1e-5);
        let second = measure_and_collapse(&mut sv, 0, &mut rng, &mut creg, 1)
            .expect("qubit 0 and creg index 1 are in range");
        assert_eq!(first, second);
    }

    #[test]
    fn t07_conditional_unmet_leaves_state_unchanged() {
        let mut sv = StateVector::new_zero_state(2).expect("2 is a valid qubit count");
        let mut rng = LcgRng::new(9);
        let mut creg = ClassicalRegister::new(1);
        // Measure |0> ⇒ bit 0 = false.
        measure_and_collapse(&mut sv, 0, &mut rng, &mut creg, 0)
            .expect("qubit 0 and creg index 0 are in range");
        let before = sv.amps.clone();
        // Predicate requires bit 0 == true, which is unmet.
        apply_if(&mut sv, &creg, &[(0, true)], &gate_x(), 1)
            .expect("bit 0 is measured and qubit 1 is in range");
        for (a, b) in sv.amps.iter().zip(before.iter()) {
            assert!((a - b).norm() < 1e-6);
        }
    }

    #[test]
    fn t08_apply_if_unmeasured_bit_errors() {
        let mut sv = StateVector::new_zero_state(2).expect("2 is a valid qubit count");
        let creg = ClassicalRegister::new(2);
        // creg bit 0 is unmeasured (None) ⇒ error.
        let res = apply_if(&mut sv, &creg, &[(0, true)], &gate_x(), 1);
        assert!(res.is_err());
    }

    #[test]
    fn t09_out_of_range_qubit_errors() {
        let mut sv = StateVector::new_zero_state(2).expect("2 is a valid qubit count");
        let mut rng = LcgRng::new(1);
        let mut creg = ClassicalRegister::new(1);
        assert!(measure_and_collapse(&mut sv, 9, &mut rng, &mut creg, 0).is_err());
    }

    #[test]
    fn t10_out_of_range_creg_idx_errors() {
        let mut sv = StateVector::new_zero_state(2).expect("2 is a valid qubit count");
        let mut rng = LcgRng::new(1);
        let mut creg = ClassicalRegister::new(1);
        assert!(measure_and_collapse(&mut sv, 0, &mut rng, &mut creg, 5).is_err());
    }

    #[test]
    fn t11_run_mixed_op_list() {
        // q0 = |1> via X; measure into bit 0; conditional X on q1 iff bit 0 == 1.
        let ops = vec![
            MidCircuitOp::Gate1q {
                gate: gate_x(),
                qubit: 0,
            },
            MidCircuitOp::Measure {
                qubit: 0,
                creg_idx: 0,
            },
            MidCircuitOp::Conditional {
                predicate: vec![(0, true)],
                gate: gate_x(),
                target: 1,
            },
        ];
        let mut rng = LcgRng::new(42);
        let (state, creg) = run(&ops, 2, &mut rng).expect("valid 3-op program with 2 qubits");
        assert_eq!(
            creg.get(0)
                .expect("creg index 0 is within run-computed bounds"),
            Some(true)
        );
        // Both qubits |1> ⇒ index 0b11 = 3.
        assert!((state.amps[3].norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn t12_measure_deterministic_forces_outcome() {
        let mut sv = StateVector::new_zero_state(1).expect("1 is a valid qubit count");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("qubit 0 is within 1-qubit state");
        measure_deterministic(&mut sv, 0, true).expect("|+> has nonzero probability for outcome 1");
        assert!((sv.amps[1].norm() - 1.0).abs() < 1e-5);
        assert!(sv.amps[0].norm() < 1e-5);
        assert!((sv.norm_sq() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn t13_measure_deterministic_zero_prob_errors() {
        // |0> forced to outcome 1 has zero probability ⇒ error.
        let mut sv = StateVector::new_zero_state(1).expect("1 is a valid qubit count");
        assert!(measure_deterministic(&mut sv, 0, true).is_err());
    }

    #[test]
    fn t14_classical_register_get_set_range() {
        let mut creg = ClassicalRegister::new(3);
        assert_eq!(creg.len(), 3);
        assert!(!creg.is_empty());
        creg.set(1, true).expect("index 1 is within 3-bit register");
        assert_eq!(
            creg.get(1).expect("index 1 is within 3-bit register"),
            Some(true)
        );
        assert_eq!(creg.get(0).expect("index 0 is within 3-bit register"), None);
        assert!(creg.set(5, true).is_err());
        assert!(creg.get(5).is_err());
    }

    #[test]
    fn t15_apply_if_multi_bit_predicate() {
        // Two control bits; gate applies only when both are true.
        let mut sv = StateVector::new_zero_state(1).expect("1 is a valid qubit count");
        let mut creg = ClassicalRegister::new(2);
        creg.set(0, true).expect("index 0 is within 2-bit register");
        creg.set(1, false)
            .expect("index 1 is within 2-bit register");
        // Predicate (0,true) AND (1,true): unmet (bit 1 is false) ⇒ no-op.
        apply_if(&mut sv, &creg, &[(0, true), (1, true)], &gate_x(), 0)
            .expect("bits 0 and 1 are measured and qubit 0 is in range");
        assert!((sv.amps[0].norm() - 1.0).abs() < 1e-5);
        // Now set bit 1 true ⇒ predicate met ⇒ X flips |0> -> |1>.
        creg.set(1, true).expect("index 1 is within 2-bit register");
        apply_if(&mut sv, &creg, &[(0, true), (1, true)], &gate_x(), 0)
            .expect("bits 0 and 1 are measured and qubit 0 is in range");
        assert!((sv.amps[1].norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn t16_apply_if_target_out_of_range_errors() {
        let mut sv = StateVector::new_zero_state(1).expect("1 is a valid qubit count");
        let mut creg = ClassicalRegister::new(1);
        creg.set(0, true).expect("index 0 is within 1-bit register");
        assert!(apply_if(&mut sv, &creg, &[(0, true)], &gate_x(), 9).is_err());
    }

    #[test]
    fn t17_run_no_measure_pure_gates() {
        // Pure gate program: H on q0 then CNOT-like via two ops is not 2q here,
        // so just verify a single-qubit superposition is produced.
        let ops = vec![MidCircuitOp::Gate1q {
            gate: gate_h(),
            qubit: 0,
        }];
        let mut rng = LcgRng::new(11);
        let (state, creg) = run(&ops, 1, &mut rng).expect("valid gate-only program with 1 qubit");
        let inv = std::f32::consts::FRAC_1_SQRT_2;
        assert!((state.amps[0].re - inv).abs() < 1e-5);
        assert!((state.amps[1].re - inv).abs() < 1e-5);
        // Register defaults to length 1, unmeasured.
        assert_eq!(creg.get(0).expect("register defaults to length 1"), None);
    }

    #[test]
    fn t18_run_invalid_qubits_errors() {
        let ops: Vec<MidCircuitOp> = vec![];
        let mut rng = LcgRng::new(1);
        assert!(run(&ops, 0, &mut rng).is_err());
    }
}
