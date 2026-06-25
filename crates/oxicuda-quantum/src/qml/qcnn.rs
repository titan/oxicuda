//! Quantum Convolutional Neural Network (QCNN) with translation-invariant layers.
//!
//! Reference: Cong, Choi & Lukin, "Quantum convolutional neural networks",
//! Nat. Phys. 15, 1273 (2019).
//!
//! A QCNN alternates **convolution** and **pooling** layers, mirroring a
//! classical CNN but acting on a quantum state:
//!
//! * A **convolution** layer applies the *same* parametrized two-qubit unitary
//!   `W(θ_conv)` to a translationally-invariant pattern of neighboring qubit
//!   pairs (a brick-wall of even pairs then odd pairs). Sharing one parameter
//!   block across all pairs is exactly the weight-sharing / translation
//!   invariance that gives a CNN its inductive bias and keeps the parameter count
//!   `O(1)` per layer rather than `O(n)`.
//! * A **pooling** layer applies a parametrized two-qubit `P(θ_pool)` to each
//!   `(kept, discarded)` pair and then *removes* the discarded qubit from the
//!   active set, halving the width — the quantum analogue of spatial pooling /
//!   coarse-graining. (Here pooling is realized coherently by an entangling
//!   block followed by tracing the discarded qubit out of the *active index set*;
//!   the state vector keeps all qubits but later layers only act on the active
//!   ones.)
//!
//! The hierarchy contracts `n → n/2 → n/4 → … → 1` active qubits; the network
//! output is the Pauli-`Z` expectation on the single surviving qubit, used as a
//! binary classifier score in `[-1, +1]`. Parameters are trained by the
//! parameter-shift rule.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::controlled::apply_cnot;
use crate::gates::parametric::{gate_ry, gate_rz};
use crate::pauli::expval::expectation_value;
use crate::pauli::hamiltonian::Hamiltonian;
use crate::pauli::pauli_string::PauliOp;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Number of trainable angles in one convolution block `W(θ_conv)`.
pub const CONV_PARAMS: usize = 6;
/// Number of trainable angles in one pooling block `P(θ_pool)`.
pub const POOL_PARAMS: usize = 6;

/// A translation-invariant QCNN classifier.
#[derive(Debug, Clone)]
pub struct Qcnn {
    n_qubits: usize,
    /// Number of (conv, pool) stages = `log2(n_qubits)`.
    n_stages: usize,
}

impl Qcnn {
    /// Construct a QCNN on `n_qubits`, which must be a power of two `≥ 2`.
    ///
    /// # Errors
    /// Returns [`QuantumError::InvalidQubitCount`] if `n_qubits` is not a power
    /// of two in `[2, 16]`.
    pub fn new(n_qubits: usize) -> QuantumResult<Self> {
        if !(2..=16).contains(&n_qubits) || !n_qubits.is_power_of_two() {
            return Err(QuantumError::InvalidQubitCount { n: n_qubits });
        }
        let n_stages = n_qubits.trailing_zeros() as usize;
        Ok(Self { n_qubits, n_stages })
    }

    /// Number of qubits.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Total number of trainable parameters: `(CONV_PARAMS + POOL_PARAMS)` per stage.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.n_stages * (CONV_PARAMS + POOL_PARAMS)
    }

    /// Apply a parametrized two-qubit block to `(a, b)` using 6 angles:
    /// `Ry,Rz` on each qubit around a single CNOT entangler.
    fn two_qubit_block(sv: &mut StateVector, a: usize, b: usize, p: &[f32]) -> QuantumResult<()> {
        apply_1q_inplace(sv, a, &gate_ry(p[0]))?;
        apply_1q_inplace(sv, a, &gate_rz(p[1]))?;
        apply_1q_inplace(sv, b, &gate_ry(p[2]))?;
        apply_1q_inplace(sv, b, &gate_rz(p[3]))?;
        apply_cnot(sv, a, b)?;
        apply_1q_inplace(sv, a, &gate_ry(p[4]))?;
        apply_1q_inplace(sv, b, &gate_ry(p[5]))?;
        Ok(())
    }

    /// Run the QCNN forward on an already-embedded input state and return the
    /// classifier score `⟨Z⟩` on the final surviving qubit.
    ///
    /// # Errors
    /// Returns [`QuantumError::DimensionMismatch`] if the input width or the
    /// parameter count is wrong; propagates gate/expectation errors.
    pub fn forward(&self, input: &StateVector, params: &[f32]) -> QuantumResult<f32> {
        if input.n_qubits != self.n_qubits {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_qubits,
                got: input.n_qubits,
            });
        }
        if params.len() != self.n_params() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_params(),
                got: params.len(),
            });
        }

        let mut sv = input.clone();
        // Active qubit indices (physical qubit numbers still alive).
        let mut active: Vec<usize> = (0..self.n_qubits).collect();
        let block = CONV_PARAMS + POOL_PARAMS;

        for stage in 0..self.n_stages {
            let conv_p = &params[stage * block..stage * block + CONV_PARAMS];
            let pool_p =
                &params[stage * block + CONV_PARAMS..stage * block + CONV_PARAMS + POOL_PARAMS];

            // --- Convolution: translation-invariant brick-wall over active pairs. ---
            // Even pairs (0,1),(2,3),...
            let mut i = 0;
            while i + 1 < active.len() {
                Self::two_qubit_block(&mut sv, active[i], active[i + 1], conv_p)?;
                i += 2;
            }
            // Odd pairs (1,2),(3,4),... for overlap (only if >2 active).
            if active.len() > 2 {
                let mut j = 1;
                while j + 1 < active.len() {
                    Self::two_qubit_block(&mut sv, active[j], active[j + 1], conv_p)?;
                    j += 2;
                }
            }

            // --- Pooling: entangle (keep, discard) pairs, then drop discards. ---
            let mut kept: Vec<usize> = Vec::with_capacity(active.len() / 2);
            let mut k = 0;
            while k + 1 < active.len() {
                let keep = active[k];
                let discard = active[k + 1];
                Self::two_qubit_block(&mut sv, keep, discard, pool_p)?;
                kept.push(keep);
                k += 2;
            }
            // Odd leftover qubit (only when active.len() is odd) is kept as-is;
            // for power-of-two widths this never triggers.
            if active.len() % 2 == 1 {
                kept.push(active[active.len() - 1]);
            }
            active = kept;
        }

        // Read out ⟨Z⟩ on the single surviving qubit.
        let survivor = active[0];
        let mut ops = vec![PauliOp::I; self.n_qubits];
        ops[survivor] = PauliOp::Z;
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, ops);
        expectation_value(&sv, &ham)
    }

    /// Parameter-shift gradient of `forward` w.r.t. each parameter, for a fixed
    /// input state.
    ///
    /// # Errors
    /// Propagates `forward` errors.
    pub fn gradient(&self, input: &StateVector, params: &[f32]) -> QuantumResult<Vec<f32>> {
        let shift = std::f32::consts::FRAC_PI_2;
        let mut grad = vec![0.0_f32; params.len()];
        for p in 0..params.len() {
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[p] += shift;
            pm[p] -= shift;
            let fp = self.forward(input, &pp)?;
            let fm = self.forward(input, &pm)?;
            grad[p] = 0.5 * (fp - fm);
        }
        Ok(grad)
    }

    /// Train the QCNN as a binary classifier by minimizing the mean-squared
    /// error between the score `⟨Z⟩` and `±1` labels via gradient descent.
    ///
    /// Returns the loss at every iteration (length `iters + 1`).
    ///
    /// # Errors
    /// Returns [`QuantumError::DimensionMismatch`] on inconsistent input widths,
    /// parameter length, or a label/data length mismatch; propagates gate errors.
    pub fn train(
        &self,
        inputs: &[StateVector],
        labels: &[f32],
        params: &mut [f32],
        iters: usize,
        lr: f32,
    ) -> QuantumResult<Vec<f32>> {
        if inputs.is_empty() {
            return Err(QuantumError::EmptyInput);
        }
        if labels.len() != inputs.len() {
            return Err(QuantumError::DimensionMismatch {
                expected: inputs.len(),
                got: labels.len(),
            });
        }
        if params.len() != self.n_params() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_params(),
                got: params.len(),
            });
        }

        let mse = |p: &[f32], me: &Self| -> QuantumResult<f32> {
            let mut acc = 0.0_f32;
            for (x, &y) in inputs.iter().zip(labels.iter()) {
                let s = me.forward(x, p)?;
                acc += (s - y) * (s - y);
            }
            Ok(acc / inputs.len() as f32)
        };

        let mut history = Vec::with_capacity(iters + 1);
        history.push(mse(params, self)?);

        for _ in 0..iters {
            // Aggregate gradient of MSE: ∂/∂θ (1/N) Σ (s-y)² = (2/N) Σ (s-y) ∂s/∂θ.
            let mut grad = vec![0.0_f32; params.len()];
            for (x, &y) in inputs.iter().zip(labels.iter()) {
                let s = self.forward(x, params)?;
                let g = self.gradient(x, params)?;
                let coeff = 2.0 * (s - y) / inputs.len() as f32;
                for (gg, gi) in grad.iter_mut().zip(g.iter()) {
                    *gg += coeff * gi;
                }
            }
            for (p, g) in params.iter_mut().zip(grad.iter()) {
                *p -= lr * g;
            }
            history.push(mse(params, self)?);
        }
        Ok(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::angle::angle_embedding;
    use crate::handle::LcgRng;

    #[test]
    fn rejects_non_power_of_two() {
        assert!(Qcnn::new(3).is_err());
        assert!(Qcnn::new(6).is_err());
        assert!(Qcnn::new(1).is_err());
        assert!(Qcnn::new(4).is_ok());
        assert!(Qcnn::new(8).is_ok());
    }

    #[test]
    fn param_count_matches_stages() {
        let q = Qcnn::new(4).expect("valid");
        // 2 stages × 12 params = 24.
        assert_eq!(q.n_params(), 24);
        let q8 = Qcnn::new(8).expect("valid");
        // 3 stages × 12 = 36.
        assert_eq!(q8.n_params(), 36);
    }

    #[test]
    fn forward_score_in_unit_interval() {
        let q = Qcnn::new(4).expect("valid");
        let mut rng = LcgRng::new(7);
        let params: Vec<f32> = (0..q.n_params()).map(|_| rng.next_normal() * 0.3).collect();
        let input = angle_embedding(&[0.5, 1.0, 0.2, 0.8]).expect("embed");
        let s = q.forward(&input, &params).expect("forward");
        assert!((-1.0001..=1.0001).contains(&s), "score={s}");
    }

    #[test]
    fn identity_params_pass_through_z_expectation() {
        // All-zero parameters → conv/pool blocks are Ry(0)Rz(0)·CNOT·Ry(0).
        // With |0000⟩ input the survivor stays |0⟩ on the Z basis under CNOTs,
        // so ⟨Z⟩ on the survivor must be +1.
        let q = Qcnn::new(4).expect("valid");
        let params = vec![0.0_f32; q.n_params()];
        let input = StateVector::new_zero_state(4).expect("zero");
        let s = q.forward(&input, &params).expect("forward");
        assert!((s - 1.0).abs() < 1e-4, "score={s}");
    }

    #[test]
    fn gradient_has_correct_length_and_is_finite() {
        let q = Qcnn::new(4).expect("valid");
        let mut rng = LcgRng::new(3);
        let params: Vec<f32> = (0..q.n_params()).map(|_| rng.next_normal() * 0.5).collect();
        let input = angle_embedding(&[0.1, 0.9, -0.3, 0.4]).expect("embed");
        let g = q.gradient(&input, &params).expect("grad");
        assert_eq!(g.len(), q.n_params());
        assert!(g.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn training_reduces_loss_on_separable_data() {
        // Two classes encoded by very different angle patterns; QCNN should learn
        // to separate them (MSE to ±1 labels should drop).
        let q = Qcnn::new(4).expect("valid");
        let class_a = angle_embedding(&[0.1, 0.1, 0.1, 0.1]).expect("a");
        let class_b = angle_embedding(&[3.0, 3.0, 3.0, 3.0]).expect("b");
        let inputs = vec![class_a.clone(), class_b.clone(), class_a, class_b];
        let labels = vec![1.0_f32, -1.0, 1.0, -1.0];
        let mut rng = LcgRng::new(2024);
        let mut params: Vec<f32> = (0..q.n_params()).map(|_| rng.next_normal() * 0.2).collect();
        let history = q
            .train(&inputs, &labels, &mut params, 30, 0.25)
            .expect("train");
        let first = history[0];
        let last = *history.last().expect("history");
        assert!(last < first, "loss did not decrease: {first} → {last}");
    }
}
